$shader_path = Convert-Path "$PSScriptRoot\..\resources\shaders\"
$compiler_path = 'dxc'

Get-ChildItem -Path $shader_path -Filter *.hlsl -Recurse -File | ForEach-Object {
    $file_path = $_.FullName
    $filename = $_.BaseName

    $shader_type = ''
    If($filename.EndsWith('vert')) {
        $shader_type = 'vs_6_0'
    }
    ElseIf ($filename.EndsWith('frag')) {
        $shader_type = 'ps_6_0'
    }
    Else {
        continue
    }

    $file_out = [IO.Path]::Combine($shader_path, 'compiled', $filename)
    $file_out = $file_out + '.spv'
    & $compiler_path $file_path -T $shader_type -E main -Fo $file_out -spirv
}