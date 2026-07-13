Set WshShell = CreateObject("WScript.Shell")
Set FileSystem = CreateObject("Scripting.FileSystemObject")

ProjectRoot = FileSystem.GetParentFolderName(FileSystem.GetParentFolderName(WScript.ScriptFullName))
WshShell.CurrentDirectory = ProjectRoot
WshShell.Run "pythonw.exe -B " & Chr(34) & ProjectRoot & "\run.py" & Chr(34), 0, False
