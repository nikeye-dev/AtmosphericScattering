#include "common.hlsl"

struct PS_INPUT
{
    float4 position: SV_Position;
    float4 fragColor: COLOR0;
    float3 normal: TEXCOORD0;
    float4 worldPos: TEXCOORD1;
};

struct PS_OUTPUT
{
    float4 color: SV_Target;
};


//Test
struct FSampleData
{
	float4 planetPos;
	float planetRadius;
	float atmosphereThickness;
	float sampleCount;
	float sampleCountLight;
	float scale;

	float4 lightDir;
	float4 lightIntensity;
};

ConstantBuffer<FSampleData> sampleData : register(b3);

PS_OUTPUT main(PS_INPUT input)
{
    PS_OUTPUT result;

//     float diffuse = saturate(dot(input.normal, sampleData.lightDir.xyz));
//     result.color = float4(input.normal, 1.0);
//     result.color = diffuse;

    result.color = float4(0.0, 0.9, 0.1, 1.0);

    return result;
}