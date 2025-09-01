# Set the folder path where your YOLO .txt files are located
$folderPath = "G:\sandbox_ai\!datasets"

# Get all .txt files in the folder
Get-ChildItem -Path $folderPath -Filter "*.txt" | ForEach-Object {
    $file = $_.FullName

    # Read each line, update first number if it's 1
    $updated = Get-Content $file | ForEach-Object {
        $parts = $_ -split '\s+'
        if ($parts[0] -eq '0') {
            $parts[0] = '1'
        }
        $parts -join ' '
    }

    # Overwrite the file with the updated content
    Set-Content -Path $file -Value $updated
}
