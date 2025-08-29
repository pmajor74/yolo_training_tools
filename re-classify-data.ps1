# Set the folder path where your YOLO .txt files are located
$folderPath = "G:\sandbox_ai\yolo_training_tools\dataset_form8_11\form8"

# Get all .txt files in the folder
Get-ChildItem -Path $folderPath -Filter "*.txt" | ForEach-Object {
    $file = $_.FullName

    # Read each line, update first number if it's 1
    $updated = Get-Content $file | ForEach-Object {
        $parts = $_ -split '\s+'
        if ($parts[0] -eq '1') {
            $parts[0] = '0'
        }
        $parts -join ' '
    }

    # Overwrite the file with the updated content
    Set-Content -Path $file -Value $updated
}
