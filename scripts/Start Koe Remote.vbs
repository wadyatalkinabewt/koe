Set WshShell = CreateObject("WScript.Shell")
WshShell.Run Chr(34) & WScript.ScriptFullName & "\..\Start Koe Remote.bat" & Chr(34), 0
Set WshShell = Nothing
