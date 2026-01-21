Set WshShell = CreateObject("WScript.Shell")
WshShell.Run Chr(34) & WScript.ScriptFullName & "\..\Stop Koe.bat" & Chr(34), 0
Set WshShell = Nothing
