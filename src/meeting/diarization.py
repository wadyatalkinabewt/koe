"""
Speaker diarization using pyannote-audio.
Identifies different speakers in audio and can match to known voices.

Setup required:
1. pip install pyannote.audio
2. Accept license at https://huggingface.co/pyannote/speaker-diarization-3.1
3. Accept license at https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM
4. Set HF_TOKEN environment variable with your HuggingFace token
"""

import os
import time
import numpy as np
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path

# Debug logging to file
_debug_log_path = Path(__file__).parent.parent.parent / "meeting_debug.log"
def _dlog(msg: str):
    with open(_debug_log_path, "a", encoding="utf-8") as f:
        f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
    print(msg)

# Try to import pyannote - it's optional
PYANNOTE_AVAILABLE = False
try:
    from pyannote.audio import Pipeline, Model, Inference
    from pyannote.audio.pipelines.utils.hook import ProgressHook
    import torch
    import torch.nn.functional as F
    PYANNOTE_AVAILABLE = True
except ImportError:
    pass


@dataclass
class SpeakerSegment:
    """A segment of audio attributed to a speaker."""
    start: float      # Start time in seconds
    end: float        # End time in seconds
    speaker: str      # Speaker label (e.g., "SPEAKER_00" or "Callum")
    confidence: float = 1.0


class SpeakerDiarizer:
    """
    Identifies different speakers in audio using pyannote-audio.

    Usage:
        diarizer = SpeakerDiarizer()
        if diarizer.is_available():
            segments = diarizer.diarize(audio_data, sample_rate=16000)
    """

    def __init__(self, device: str = "cuda"):
        self._pipeline = None
        self._embedding_model = None  # For speaker fingerprinting
        self._embedding_inference = None
        self._device = device
        self._known_speakers: Dict[str, np.ndarray] = {}  # name -> embedding (enrolled speakers)
        self._embeddings_dir = Path(__file__).parent.parent.parent / "speaker_embeddings"
        self._similarity_threshold = 0.25  # Cosine similarity threshold for matching (lowered for cross-chunk variance)

        # Session speaker tracking (for cross-chunk consistency)
        self._session_speakers: Dict[str, np.ndarray] = {}  # "Speaker 1" -> embedding
        self._session_speaker_counter = 0
        self._session_embedding_counts: Dict[str, int] = {}  # Track how many embeddings contributed
        self._enrolled_seen_this_session: set = set()  # Track which enrolled speakers have been seen
        self._last_active_session_speaker: Optional[str] = None  # Most recently assigned session speaker

        # Session persistence (survives server restarts)
        self._session_state_file = Path(__file__).parent.parent.parent / ".session_state.npz"
        self._session_start_time: Optional[float] = None  # When this session started
        self._session_max_age = 3600  # Session state valid for 1 hour

        # Adaptive enrollment settings
        self._adaptive_threshold = 0.6  # Only update if similarity > this (high confidence)
        self._adaptive_rate = 0.05  # 5% new, 95% old - very slow adaptation
        self._enrolled_updated: Dict[str, bool] = {}  # Track which speakers were updated this session

    def is_available(self) -> bool:
        """Check if pyannote-audio is available and configured."""
        if not PYANNOTE_AVAILABLE:
            return False

        # Check for HuggingFace token
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not hf_token:
            return False

        return True

    def load(self) -> bool:
        """Load the diarization pipeline. Returns True if successful."""
        if not PYANNOTE_AVAILABLE:
            _dlog("[Diarization] pyannote.audio not installed. Install with: pip install pyannote.audio")
            return False

        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not hf_token:
            _dlog("[Diarization] HuggingFace token not found. Set HF_TOKEN environment variable.")
            _dlog("[Diarization] Get token at: https://huggingface.co/settings/tokens")
            _dlog("[Diarization] Accept license at: https://huggingface.co/pyannote/speaker-diarization-3.1")
            return False

        try:
            _dlog("[Diarization] Loading pyannote speaker-diarization pipeline...")
            self._pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=hf_token
            )

            # Move to GPU if available
            device = torch.device("cuda" if self._device == "cuda" and torch.cuda.is_available() else "cpu")
            self._pipeline.to(device)
            _dlog(f"[Diarization] Pipeline loaded on {'GPU' if device.type == 'cuda' else 'CPU'}")

            # Load embedding model for speaker fingerprinting
            _dlog("[Diarization] Loading embedding model for speaker fingerprinting...")
            self._embedding_model = Model.from_pretrained(
                "pyannote/wespeaker-voxceleb-resnet34-LM",
                token=hf_token
            )
            self._embedding_model.to(device)
            self._embedding_inference = Inference(self._embedding_model, window="whole")
            _dlog("[Diarization] Embedding model loaded")

            # Load known speaker embeddings if they exist
            self._load_known_speakers()

            # Try to restore session state (for crash recovery)
            if self._load_session_state():
                _dlog("[Diarization] Recovered previous session state")

            return True

        except Exception as e:
            _dlog(f"[Diarization] Failed to load pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_known_speakers(self):
        """Load pre-enrolled speaker embeddings."""
        _dlog(f"[Diarization] Loading known speakers from: {self._embeddings_dir}")

        if not self._embeddings_dir.exists():
            _dlog(f"[Diarization] Embeddings directory does not exist: {self._embeddings_dir}")
            return

        npy_files = list(self._embeddings_dir.glob("*.npy"))
        _dlog(f"[Diarization] Found {len(npy_files)} .npy files: {[f.stem for f in npy_files]}")

        for emb_file in npy_files:
            name = emb_file.stem
            try:
                embedding = np.load(emb_file)
                self._known_speakers[name] = embedding
                _dlog(f"[Diarization] Loaded speaker embedding: {name} (shape={embedding.shape}, norm={np.linalg.norm(embedding):.4f})")
            except Exception as e:
                _dlog(f"[Diarization] Failed to load {name}: {e}")

        _dlog(f"[Diarization] Total known speakers loaded: {list(self._known_speakers.keys())}")

    def diarize(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        num_speakers: Optional[int] = None,
        min_speakers: int = 1,
        max_speakers: int = 8
    ) -> List[SpeakerSegment]:
        """
        Identify speakers in audio.

        Args:
            audio: Audio data as int16 numpy array
            sample_rate: Sample rate of audio
            num_speakers: Exact number of speakers (if known)
            min_speakers: Minimum expected speakers
            max_speakers: Maximum expected speakers

        Returns:
            List of SpeakerSegment with start/end times and speaker labels
        """
        if self._pipeline is None:
            if not self.load():
                return []

        try:
            # Convert to float32 for pyannote
            audio_float = audio.astype(np.float32) / 32768.0

            # Check audio level
            rms = np.sqrt(np.mean(audio_float ** 2))
            max_val = np.max(np.abs(audio_float))
            _dlog(f"[Diarization] Audio stats: RMS={rms:.4f}, max={max_val:.4f}, len={len(audio_float)}")

            # Create waveform tensor (channels, samples)
            waveform = torch.from_numpy(audio_float).unsqueeze(0)

            # Create input dict
            input_dict = {
                "waveform": waveform,
                "sample_rate": sample_rate
            }

            # Run diarization
            _dlog(f"[Diarization] Running pipeline with min_speakers={min_speakers}, max_speakers={max_speakers}")
            if num_speakers:
                diarization = self._pipeline(input_dict, num_speakers=num_speakers)
            else:
                diarization = self._pipeline(
                    input_dict,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers
                )

            _dlog(f"[Diarization] Pipeline result type: {type(diarization)}")

            # New pyannote API returns DiarizeOutput - extract the Annotation
            annotation = getattr(diarization, 'speaker_diarization', diarization)
            _dlog(f"[Diarization] Annotation type: {type(annotation)}")

            # Get all tracks
            track_list = list(annotation.itertracks(yield_label=True))
            _dlog(f"[Diarization] Raw tracks found: {len(track_list)}")

            # Get unique speaker labels from this chunk
            speaker_labels = annotation.labels()
            _dlog(f"[Diarization] Speaker labels in chunk: {speaker_labels}")

            # For each speaker, extract their audio and get wespeaker embedding
            # This ensures we use the SAME embedding model as enrollment
            chunk_to_session: Dict[str, str] = {}

            for chunk_label in speaker_labels:
                # Collect all audio segments for this speaker
                speaker_audio_segments = []
                for turn, _, spk in track_list:
                    if spk == chunk_label:
                        start_sample = int(turn.start * sample_rate)
                        end_sample = int(turn.end * sample_rate)
                        if end_sample > start_sample and end_sample <= len(audio):
                            speaker_audio_segments.append(audio[start_sample:end_sample])

                if not speaker_audio_segments:
                    continue

                # Concatenate all segments for this speaker
                speaker_audio = np.concatenate(speaker_audio_segments)

                # Need at least 0.5s of audio for reliable embedding
                if len(speaker_audio) < sample_rate * 0.5:
                    _dlog(f"[Diarization] {chunk_label}: Only {len(speaker_audio)/sample_rate:.2f}s audio, too short for embedding")
                    continue

                # Extract embedding using our wespeaker model (same as enrollment)
                emb = self._extract_embedding(speaker_audio, sample_rate)
                if emb is None:
                    _dlog(f"[Diarization] {chunk_label}: Failed to extract embedding")
                    continue

                # First check enrolled speakers
                matched_enrolled = self._match_embedding_to_known(emb)
                if matched_enrolled:
                    chunk_to_session[chunk_label] = matched_enrolled
                    self._enrolled_seen_this_session.add(matched_enrolled)  # Track for max_speakers
                    _dlog(f"[Diarization] {chunk_label} -> enrolled: {matched_enrolled}")
                    continue

                # Then check session speakers (cross-chunk tracking)
                matched_session = self._match_embedding_to_session(emb)
                if matched_session:
                    chunk_to_session[chunk_label] = matched_session
                    self._last_active_session_speaker = matched_session  # Track for fallback
                    _dlog(f"[Diarization] {chunk_label} -> session: {matched_session}")
                    # Update session embedding with running average for better matching
                    self._update_session_embedding(matched_session, emb)
                else:
                    # Enforce max_speakers: count BOTH enrolled and unknown speakers
                    total_speakers = len(self._enrolled_seen_this_session) + len(self._session_speakers)
                    if total_speakers >= max_speakers:
                        closest = self._find_closest_session_speaker(emb)
                        if closest:
                            chunk_to_session[chunk_label] = closest
                            self._update_session_embedding(closest, emb)
                            _dlog(f"[Diarization] {chunk_label} -> MERGED (total={total_speakers}, max={max_speakers}): {closest}")
                            continue  # Skip creating new speaker

                    # New speaker - add to session
                    self._session_speaker_counter += 1
                    new_label = f"Speaker {self._session_speaker_counter}"
                    self._session_speakers[new_label] = emb
                    self._session_embedding_counts[new_label] = 1
                    chunk_to_session[chunk_label] = new_label
                    self._last_active_session_speaker = new_label  # Track for fallback
                    # Set session start time on first speaker
                    if self._session_start_time is None:
                        self._session_start_time = time.time()
                    # Save session state for crash recovery
                    self._save_session_state()
                    _dlog(f"[Diarization] {chunk_label} -> NEW: {new_label}")

            # Convert to segments with consistent labels
            segments = []
            for turn, _, speaker in track_list:
                # Use session-consistent label
                consistent_label = chunk_to_session.get(speaker)

                if consistent_label is None:
                    # Unmapped speaker (embedding extraction failed) - assign to existing session speaker
                    # This prevents raw SPEAKER_XX labels leaking through AND prevents speaker fragmentation

                    # Prefer: last active session speaker > any session speaker > new speaker
                    if self._last_active_session_speaker and self._last_active_session_speaker in self._session_speakers:
                        # Assign to last active (most likely the same person speaking)
                        consistent_label = self._last_active_session_speaker
                        chunk_to_session[speaker] = consistent_label
                        _dlog(f"[Diarization] {speaker} -> fallback REUSE (last_active): {consistent_label}")
                    elif self._session_speakers:
                        # Assign to first session speaker (better than creating new)
                        consistent_label = next(iter(self._session_speakers.keys()))
                        chunk_to_session[speaker] = consistent_label
                        _dlog(f"[Diarization] {speaker} -> fallback REUSE (first): {consistent_label}")
                    else:
                        # No session speakers exist yet - must create first one
                        self._session_speaker_counter += 1
                        consistent_label = f"Speaker {self._session_speaker_counter}"
                        self._session_speakers[consistent_label] = np.zeros(256)  # Placeholder embedding
                        self._session_embedding_counts[consistent_label] = 0
                        chunk_to_session[speaker] = consistent_label
                        self._last_active_session_speaker = consistent_label
                        # Set session start time on first speaker
                        if self._session_start_time is None:
                            self._session_start_time = time.time()
                        # Save session state for crash recovery
                        self._save_session_state()
                        _dlog(f"[Diarization] {speaker} -> fallback NEW (first): {consistent_label}")

                segments.append(SpeakerSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker=consistent_label
                ))

            _dlog(f"[Diarization] Returning {len(segments)} segments")
            return segments

        except Exception as e:
            _dlog(f"[Diarization] Error: {e}")
            return []

    def _extract_embedding(self, audio: np.ndarray, sample_rate: int = 16000) -> Optional[np.ndarray]:
        """
        Extract a voice embedding from audio.

        Args:
            audio: Audio data as int16 numpy array
            sample_rate: Sample rate of audio

        Returns:
            512-dimensional embedding vector, or None on error
        """
        if self._embedding_inference is None:
            return None

        try:
            # Convert to float32
            audio_float = audio.astype(np.float32) / 32768.0

            # Create waveform tensor (channels, samples)
            waveform = torch.from_numpy(audio_float).unsqueeze(0)

            # Create input dict
            input_dict = {
                "waveform": waveform,
                "sample_rate": sample_rate
            }

            # Extract embedding
            embedding = self._embedding_inference(input_dict)

            # Convert to numpy and normalize
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.cpu().numpy()

            # Normalize to unit length for cosine similarity
            embedding = embedding / np.linalg.norm(embedding)

            return embedding.flatten()

        except Exception as e:
            _dlog(f"[Diarization] Error extracting embedding: {e}")
            return None

    def _match_embedding_to_known(self, embedding: np.ndarray) -> Optional[str]:
        """Match embedding to enrolled/known speakers."""
        if not self._known_speakers:
            return None

        best_match = None
        best_similarity = self._similarity_threshold

        # Log all similarity scores for debugging
        scores = []
        for name, known_emb in self._known_speakers.items():
            similarity = np.dot(embedding, known_emb)
            scores.append(f"{name}={similarity:.3f}")
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name

        _dlog(f"[Match] Enrolled scores: {', '.join(scores)} -> {'MATCHED ' + best_match if best_match else 'NO MATCH'}")

        # Adaptive learning: update enrolled embedding on high-confidence match
        if best_match and best_similarity > self._adaptive_threshold:
            self._update_enrolled_embedding(best_match, embedding, best_similarity)

        return best_match

    def _update_enrolled_embedding(self, name: str, new_embedding: np.ndarray, similarity: float):
        """
        Adaptively update enrolled speaker embedding with new sample.
        Uses very slow learning rate to gradually improve accuracy over time
        while preventing drift from occasional mismatches.
        """
        if name not in self._known_speakers:
            return

        old_embedding = self._known_speakers[name]

        # Weighted average: 95% old + 5% new (very conservative)
        updated = (old_embedding * (1 - self._adaptive_rate) + new_embedding * self._adaptive_rate)
        # Re-normalize to unit length
        updated = updated / np.linalg.norm(updated)

        self._known_speakers[name] = updated
        self._enrolled_updated[name] = True
        _dlog(f"[Adaptive] Updated {name} embedding (similarity={similarity:.3f})")

    def _match_embedding_to_session(self, embedding: np.ndarray) -> Optional[str]:
        """Match embedding to speakers seen in this session (cross-chunk tracking)."""
        if not self._session_speakers:
            return None

        best_match = None
        best_similarity = self._similarity_threshold

        # Log similarity scores for debugging
        scores = []
        for label, session_emb in self._session_speakers.items():
            similarity = np.dot(embedding, session_emb)
            scores.append(f"{label}={similarity:.3f}")
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = label

        if scores:
            _dlog(f"[Match] Session scores: {', '.join(scores[:5])}{'...' if len(scores) > 5 else ''} -> {'MATCHED ' + best_match if best_match else 'NO MATCH'}")
        return best_match

    def _update_session_embedding(self, label: str, new_embedding: np.ndarray):
        """Update session embedding with running average for better matching over time."""
        if label not in self._session_speakers:
            return

        count = self._session_embedding_counts.get(label, 1)
        old_embedding = self._session_speakers[label]

        # Running average: new_avg = (old_avg * count + new_emb) / (count + 1)
        updated = (old_embedding * count + new_embedding) / (count + 1)
        # Re-normalize to unit length
        updated = updated / np.linalg.norm(updated)

        self._session_speakers[label] = updated
        self._session_embedding_counts[label] = count + 1
        _dlog(f"[Match] Updated {label} embedding (n={count + 1})")

    def _find_closest_session_speaker(self, embedding: np.ndarray) -> Optional[str]:
        """
        Find the most similar session speaker, regardless of threshold.
        Used when max_speakers is exceeded and we need to force-merge.
        """
        if not self._session_speakers:
            return None

        best_match = None
        best_similarity = -1.0  # No threshold - just find best match

        for label, session_emb in self._session_speakers.items():
            similarity = np.dot(embedding, session_emb)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = label

        if best_match:
            _dlog(f"[MaxSpeakers] Force-merging with closest: {best_match} (sim={best_similarity:.3f})")
        return best_match

    def consolidate_session_speakers(self, similarity_threshold: float = 0.40) -> Dict[str, str]:
        """
        Merge similar session speakers to reduce fragmentation.

        Should be called AFTER meeting ends, BEFORE showing enrollment dialog.
        This catches splits that happened due to threshold misses during recording.

        Args:
            similarity_threshold: Speakers with similarity above this are merged (default 0.40)

        Returns:
            Dict mapping old speaker labels to new labels (for transcript rewriting)
            e.g., {"Speaker 5": "Speaker 3", "Speaker 7": "Speaker 3"}
        """
        if len(self._session_speakers) < 2:
            return {}

        _dlog(f"[Consolidate] Starting consolidation of {len(self._session_speakers)} session speakers")

        # Build list of (label, embedding) sorted by embedding count (most samples first)
        speakers = [(label, self._session_speakers[label], self._session_embedding_counts.get(label, 1))
                    for label in self._session_speakers]
        speakers.sort(key=lambda x: -x[2])  # Most samples first (more reliable embedding)

        # Track merges: old_label -> new_label
        merges = {}
        # Track which speakers have been merged into others
        merged_away = set()

        for i, (label1, emb1, count1) in enumerate(speakers):
            if label1 in merged_away:
                continue

            for j, (label2, emb2, count2) in enumerate(speakers[i+1:], i+1):
                if label2 in merged_away:
                    continue

                # Calculate cosine similarity (embeddings are normalized)
                similarity = np.dot(emb1, emb2)

                if similarity >= similarity_threshold:
                    # Merge label2 into label1 (label1 has more samples)
                    _dlog(f"[Consolidate] Merging '{label2}' into '{label1}' (sim={similarity:.3f}, counts={count1}/{count2})")
                    merges[label2] = label1
                    merged_away.add(label2)

                    # Update label1's embedding with combined average
                    total_count = count1 + count2
                    combined_emb = (emb1 * count1 + emb2 * count2) / total_count
                    combined_emb = combined_emb / np.linalg.norm(combined_emb)  # Re-normalize

                    self._session_speakers[label1] = combined_emb
                    self._session_embedding_counts[label1] = total_count

        # Remove merged speakers from session
        for old_label in merged_away:
            del self._session_speakers[old_label]
            self._session_embedding_counts.pop(old_label, None)

        if merges:
            _dlog(f"[Consolidate] Merged {len(merges)} speakers, {len(self._session_speakers)} remaining")
            # Save updated session state after consolidation
            self._save_session_state()
        else:
            _dlog(f"[Consolidate] No speakers merged (all below threshold {similarity_threshold})")

        return merges

    def get_unenrolled_session_speakers(self) -> Dict[str, np.ndarray]:
        """
        Get session speakers that are not enrolled (e.g., "Speaker 1", "Speaker 2").

        Call this BEFORE reset_session() to get embeddings for enrollment.

        Returns:
            Dict mapping speaker labels to their embeddings (running averages from session)
        """
        return dict(self._session_speakers)  # Return a copy

    def get_enrolled_speakers_seen(self) -> Dict[str, np.ndarray]:
        """
        Get enrolled speakers that were seen in this session.

        Call this BEFORE reset_session() to get updated embeddings.

        Returns:
            Dict mapping speaker names to their current embeddings (may have been
            updated via adaptive learning during the session)
        """
        result = {}
        for name in self._enrolled_seen_this_session:
            if name in self._known_speakers:
                result[name] = self._known_speakers[name]
        return result

    def update_enrolled_speaker(self, name: str) -> bool:
        """
        Force-save the current embedding for an enrolled speaker.

        This saves the adaptively-updated embedding immediately, rather than
        waiting for session reset.

        Args:
            name: The enrolled speaker's name

        Returns:
            True if save succeeded, False otherwise
        """
        if name not in self._known_speakers:
            _dlog(f"[Enroll] Speaker '{name}' not found in known speakers")
            return False

        embedding = self._known_speakers[name]
        embedding_file = self._embeddings_dir / f"{name}.npy"
        try:
            np.save(embedding_file, embedding)
            _dlog(f"[Enroll] Updated embedding for '{name}'")
            # Remove from updated tracking since we just saved it
            self._enrolled_updated.pop(name, None)
            return True
        except Exception as e:
            _dlog(f"[Enroll] Failed to update '{name}': {e}")
            return False

    def enroll_speaker_from_session(self, session_label: str, name: str) -> bool:
        """
        Enroll a session speaker with a given name.

        Args:
            session_label: The session label (e.g., "Speaker 1")
            name: The name to enroll as (e.g., "Sritam")

        Returns:
            True if enrollment succeeded, False otherwise
        """
        _dlog(f"[Enroll] enroll_speaker_from_session called: session_label='{session_label}', name='{name}'")
        _dlog(f"[Enroll] Available session speakers: {list(self._session_speakers.keys())}")

        if session_label not in self._session_speakers:
            _dlog(f"[Enroll] Session label '{session_label}' not found in session speakers!")
            return False

        embedding = self._session_speakers[session_label]
        _dlog(f"[Enroll] Got embedding for '{session_label}': shape={embedding.shape}, norm={np.linalg.norm(embedding):.4f}")

        # Save to file
        self._embeddings_dir.mkdir(parents=True, exist_ok=True)
        embedding_file = self._embeddings_dir / f"{name}.npy"
        _dlog(f"[Enroll] Saving to: {embedding_file}")
        try:
            np.save(embedding_file, embedding)
            # Verify file was created
            if embedding_file.exists():
                _dlog(f"[Enroll] SUCCESS: Saved '{session_label}' as '{name}' to {embedding_file}")
            else:
                _dlog(f"[Enroll] WARNING: np.save completed but file doesn't exist: {embedding_file}")

            # Add to known speakers for immediate use
            self._known_speakers[name] = embedding

            return True
        except Exception as e:
            _dlog(f"[Enroll] Failed to save '{name}': {e}")
            return False

    def enroll_speaker_with_embedding(self, name: str, embedding: np.ndarray) -> bool:
        """
        Enroll a speaker with a provided embedding.

        This is safer than enroll_speaker_from_session because it doesn't depend
        on the session speakers dictionary still being populated.

        Args:
            name: The name to enroll as (e.g., "Joe")
            embedding: The speaker embedding

        Returns:
            True if enrollment succeeded, False otherwise
        """
        _dlog(f"[Enroll] enroll_speaker_with_embedding: name='{name}', embedding shape={embedding.shape}")

        # Save to file
        self._embeddings_dir.mkdir(parents=True, exist_ok=True)
        embedding_file = self._embeddings_dir / f"{name}.npy"
        _dlog(f"[Enroll] Saving to: {embedding_file}")
        try:
            np.save(embedding_file, embedding)
            # Verify file was created
            if embedding_file.exists():
                _dlog(f"[Enroll] SUCCESS: Saved '{name}' to {embedding_file}")
            else:
                _dlog(f"[Enroll] WARNING: np.save completed but file doesn't exist: {embedding_file}")

            # Add to known speakers for immediate use
            self._known_speakers[name] = embedding

            return True
        except Exception as e:
            _dlog(f"[Enroll] Failed to save '{name}': {e}")
            return False

    def extract_user_embedding(
        self,
        audio: np.ndarray,
        user_name: str,
        sample_rate: int = 16000
    ) -> List[SpeakerSegment]:
        """
        Process mic audio for embedding extraction and timing consistency.

        Runs full diarization pipeline to:
        1. Ensure timing consistency with loopback audio (both go through same pipeline)
        2. Extract embedding from the audio
        3. Update user's enrolled embedding (adaptive learning)
        4. Mark user as seen in session (for post-meeting dialog)

        Args:
            audio: Audio data as int16 numpy array (mic audio)
            user_name: Name of the user (from config)
            sample_rate: Sample rate of audio

        Returns:
            List of SpeakerSegment with timing info (all labeled as user_name)
        """
        if self._pipeline is None:
            if not self.load():
                return []

        try:
            # Convert to float32 for pyannote
            audio_float = audio.astype(np.float32) / 32768.0

            # Create waveform tensor
            waveform = torch.from_numpy(audio_float).unsqueeze(0)
            input_dict = {"waveform": waveform, "sample_rate": sample_rate}

            # Run diarization (for timing consistency)
            _dlog(f"[UserMic] Running diarization for timing ({len(audio)/sample_rate:.1f}s)")
            diarization = self._pipeline(input_dict, min_speakers=1, max_speakers=2)

            # Extract annotation
            annotation = getattr(diarization, 'speaker_diarization', diarization)
            track_list = list(annotation.itertracks(yield_label=True))

            if not track_list:
                # No speech detected - return single segment for full audio
                duration = len(audio) / sample_rate
                _dlog(f"[UserMic] No speech detected, using full audio ({duration:.1f}s)")
                return [SpeakerSegment(start=0.0, end=duration, speaker=user_name)]

            # Extract embedding from entire mic audio (we know it's the user)
            emb = self._extract_embedding(audio, sample_rate)
            if emb is not None:
                # Update user's enrolled embedding
                if user_name in self._known_speakers:
                    # Force update (we know this is the user, not probabilistic)
                    old_emb = self._known_speakers[user_name]
                    # Use slightly higher learning rate for mic audio (we're certain it's them)
                    rate = 0.10  # 10% new, 90% old
                    updated = (old_emb * (1 - rate) + emb * rate)
                    updated = updated / np.linalg.norm(updated)
                    self._known_speakers[user_name] = updated
                    self._enrolled_updated[user_name] = True
                    _dlog(f"[UserMic] Updated {user_name}'s embedding (forced)")
                else:
                    # User not enrolled yet - enroll them now
                    self._embeddings_dir.mkdir(parents=True, exist_ok=True)
                    embedding_file = self._embeddings_dir / f"{user_name}.npy"
                    np.save(embedding_file, emb)
                    self._known_speakers[user_name] = emb
                    _dlog(f"[UserMic] Auto-enrolled {user_name} from mic audio")

                # Mark user as seen in session
                self._enrolled_seen_this_session.add(user_name)

            # Convert tracks to segments, all labeled as user_name
            segments = []
            for turn, _, _ in track_list:  # Ignore speaker label from diarization
                segments.append(SpeakerSegment(
                    start=turn.start,
                    end=turn.end,
                    speaker=user_name  # Force label as user
                ))

            _dlog(f"[UserMic] Returning {len(segments)} segments for {user_name}")
            return segments

        except Exception as e:
            _dlog(f"[UserMic] Error: {e}")
            return []

    def reset_session(self):
        """Reset session speaker tracking (call at start of new meeting)."""
        # Save any updated enrolled embeddings before resetting
        self._save_updated_embeddings()

        self._session_speakers.clear()
        self._session_embedding_counts.clear()
        self._session_speaker_counter = 0
        self._enrolled_updated.clear()
        self._enrolled_seen_this_session.clear()  # Clear enrolled speaker tracking
        self._last_active_session_speaker = None  # Clear last active speaker
        self._session_start_time = None
        # Delete session state file on reset
        self._delete_session_state()
        _dlog("[Diarization] Session reset - speaker tracking cleared")

    def _save_session_state(self):
        """Save session state to disk for persistence across server restarts."""
        if not self._session_speakers:
            return  # Nothing to save

        try:
            # Prepare data for saving
            labels = list(self._session_speakers.keys())
            embeddings = np.array([self._session_speakers[k] for k in labels])
            counts = np.array([self._session_embedding_counts.get(k, 1) for k in labels])

            np.savez(
                self._session_state_file,
                labels=labels,
                embeddings=embeddings,
                counts=counts,
                counter=self._session_speaker_counter,
                start_time=self._session_start_time or time.time(),
                last_active=self._last_active_session_speaker or ""
            )
            _dlog(f"[Session] Saved session state: {len(labels)} speakers")
        except Exception as e:
            _dlog(f"[Session] Failed to save session state: {e}")

    def _load_session_state(self) -> bool:
        """Load session state from disk if it exists and is recent.

        Returns:
            True if session state was loaded, False otherwise
        """
        if not self._session_state_file.exists():
            return False

        try:
            data = np.load(self._session_state_file, allow_pickle=True)

            # Check if session is too old
            start_time = float(data['start_time'])
            age = time.time() - start_time
            if age > self._session_max_age:
                _dlog(f"[Session] Session state too old ({age:.0f}s > {self._session_max_age}s), ignoring")
                self._delete_session_state()
                return False

            # Restore session state
            labels = list(data['labels'])
            embeddings = data['embeddings']
            counts = data['counts']

            self._session_speakers = {labels[i]: embeddings[i] for i in range(len(labels))}
            self._session_embedding_counts = {labels[i]: int(counts[i]) for i in range(len(labels))}
            self._session_speaker_counter = int(data['counter'])
            self._session_start_time = start_time
            last_active = str(data['last_active'])
            self._last_active_session_speaker = last_active if last_active else None

            _dlog(f"[Session] Restored session state: {len(labels)} speakers (age: {age:.0f}s)")
            return True

        except Exception as e:
            _dlog(f"[Session] Failed to load session state: {e}")
            self._delete_session_state()
            return False

    def _delete_session_state(self):
        """Delete session state file."""
        try:
            if self._session_state_file.exists():
                self._session_state_file.unlink()
                _dlog("[Session] Deleted session state file")
        except Exception as e:
            _dlog(f"[Session] Failed to delete session state: {e}")

    def _save_updated_embeddings(self):
        """Save enrolled embeddings that were updated during this session."""
        if not self._enrolled_updated:
            return

        for name in self._enrolled_updated:
            if name in self._known_speakers:
                embedding = self._known_speakers[name]
                embedding_file = self._embeddings_dir / f"{name}.npy"
                try:
                    np.save(embedding_file, embedding)
                    _dlog(f"[Adaptive] Saved updated embedding for {name}")
                except Exception as e:
                    _dlog(f"[Adaptive] Failed to save {name}: {e}")

    def _match_speaker(self, anonymous_label: str) -> Optional[str]:
        """Try to match anonymous speaker to known speaker based on session tracking."""
        # This is now handled by embedding matching in diarize()
        return None

    def match_segment_audio(self, segment_audio: np.ndarray, sample_rate: int = 16000) -> Optional[str]:
        """
        Match a segment of audio to a known speaker.

        Args:
            segment_audio: Audio data for this segment
            sample_rate: Sample rate

        Returns:
            Name of matched speaker, or None if no match
        """
        if not self._known_speakers:
            return None

        embedding = self._extract_embedding(segment_audio, sample_rate)
        if embedding is None:
            return None

        # Compare to all known speakers
        best_match = None
        best_similarity = self._similarity_threshold

        for name, known_embedding in self._known_speakers.items():
            # Cosine similarity (embeddings are normalized)
            similarity = np.dot(embedding, known_embedding)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = name

        if best_match:
            _dlog(f"[Diarization] Matched speaker: {best_match} (similarity: {best_similarity:.2f})")

        return best_match

    def enroll_speaker(self, name: str, audio: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        Enroll a new speaker by extracting their voice embedding.

        Args:
            name: Speaker name (e.g., "Callum")
            audio: Audio sample of the speaker (at least 3 seconds recommended)
            sample_rate: Sample rate of audio

        Returns:
            True if enrollment successful
        """
        if self._embedding_inference is None:
            if not self.load():
                _dlog("[Diarization] Cannot enroll - model not loaded")
                return False

        # Check audio length
        duration = len(audio) / sample_rate
        if duration < 1.0:
            _dlog(f"[Diarization] Audio too short ({duration:.1f}s). Need at least 1 second.")
            return False

        if duration < 3.0:
            _dlog(f"[Diarization] Warning: Audio is only {duration:.1f}s. 3+ seconds recommended for accuracy.")

        # Extract embedding
        embedding = self._extract_embedding(audio, sample_rate)
        if embedding is None:
            _dlog(f"[Diarization] Failed to extract embedding for {name}")
            return False

        # Save to file
        self._embeddings_dir.mkdir(parents=True, exist_ok=True)
        embedding_file = self._embeddings_dir / f"{name}.npy"
        np.save(embedding_file, embedding)

        # Add to known speakers
        self._known_speakers[name] = embedding

        _dlog(f"[Diarization] Enrolled speaker: {name} (saved to {embedding_file})")
        return True

    def list_enrolled_speakers(self) -> List[str]:
        """List all enrolled speaker names."""
        return list(self._known_speakers.keys())

    def remove_speaker(self, name: str) -> bool:
        """Remove an enrolled speaker."""
        if name not in self._known_speakers:
            _dlog(f"[Diarization] Speaker not found: {name}")
            return False

        del self._known_speakers[name]

        # Remove file
        embedding_file = self._embeddings_dir / f"{name}.npy"
        if embedding_file.exists():
            embedding_file.unlink()

        _dlog(f"[Diarization] Removed speaker: {name}")
        return True


# Singleton instance
_diarizer: Optional[SpeakerDiarizer] = None


def get_diarizer() -> SpeakerDiarizer:
    """Get the singleton diarizer instance."""
    global _diarizer
    if _diarizer is None:
        _diarizer = SpeakerDiarizer()
    return _diarizer
