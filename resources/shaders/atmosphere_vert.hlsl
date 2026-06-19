#include "common.hlsl"

struct VS_INPUT
{
    float3 position : POSITION;
    float3 normal : NORMAL;
    float4 color : COLOR0;
};

struct VS_PUSH_CONSTANTS
{
    float4x4 model;
};

struct VS_OUTPUT
{
    float4 position: SV_POSITION;
    float4 fragColor: COLOR0;
    float3 normal: TEXCOORD0;
    float4 worldPos: TEXCOORD1;
};

[[vk::push_constant]] VS_PUSH_CONSTANTS pcs;

VS_OUTPUT main(VS_INPUT input)
{
    VS_OUTPUT result;

    float4 pos = float4(input.position, 1.0);
    pos = mul(pcs.model, pos);

    result.worldPos = pos;

    pos = mul(transform.view, pos);
    pos = mul(transform.projection, pos);
    result.position = pos;

    result.normal = input.normal;
    result.fragColor = input.color;

    return result;
}