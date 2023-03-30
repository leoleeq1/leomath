#ifndef LEOMATH_VECTOR_H_
#define LEOMATH_VECTOR_H_

#include <immintrin.h>

#include <cassert>
#include <cmath>
#include <compare>
#include <cstdint>
#include <span>

#ifdef _MSC_VER
    #pragma warning(push)
    #pragma warning(disable : 4201)
#endif

namespace leo
{

constexpr float kPI = 3.141592654f;
constexpr float k2PI = 6.283185307f;
constexpr float kReciPI = 0.318309886f;
constexpr float kReci2PI = 0.159154943f;
constexpr float kHfPI = 1.570796327f;
constexpr float kQtPI = 0.785398163f;

constexpr uint32_t kSwizzleX = 0;
constexpr uint32_t kSwizzleY = 1;
constexpr uint32_t kSwizzleZ = 2;
constexpr uint32_t kSwizzleW = 3;

constexpr float DegToRad(float degree) noexcept
{
    return degree * (kPI / 180.0f);
}

constexpr float RadToDeg(float rad) noexcept
{
    return rad * (180.0f / kPI);
}

using FVector = __m128;

struct Vector2f
{
public:
    Vector2f() = default;

    Vector2f(const Vector2f&) = default;
    Vector2f& operator=(const Vector2f&) = default;

    Vector2f(Vector2f&&) = default;
    Vector2f& operator=(Vector2f&&) = default;

    constexpr explicit Vector2f(float f) noexcept : f32{f, f} {}
    constexpr Vector2f(float _x, float _y) noexcept : f32{_x, _y} {}
    explicit Vector2f(std::span<const float, 2> span) noexcept
        : f32{span[0], span[1]}
    {
    }

public:
    union
    {
        float f32[2];
        struct
        {
            float x, y;
        };
    };
};

struct Vector3f
{
public:
    Vector3f() = default;

    Vector3f(const Vector3f&) = default;
    Vector3f& operator=(const Vector3f&) = default;

    Vector3f(Vector3f&&) = default;
    Vector3f& operator=(Vector3f&&) = default;

    constexpr explicit Vector3f(float f) noexcept : f32{f, f, f} {}
    constexpr Vector3f(float x, float y, float z) noexcept : f32{x, y, z} {}
    explicit Vector3f(std::span<const float, 3> span) noexcept
        : f32{span[0], span[1], span[2]}
    {
    }

    explicit Vector3f(const Vector2f& v2f) noexcept : f32{v2f.x, v2f.y, 0.0f} {}
    explicit Vector3f(const Vector2f& v2f, float z) noexcept
        : f32{v2f.x, v2f.y, z}
    {
    }

    inline operator Vector2f() const noexcept { return Vector2f(x, y); }

public:
    union
    {
        float f32[3];
        struct
        {
            float x, y, z;
        };
    };
};

struct Vector4f
{
public:
    Vector4f() = default;

    Vector4f(const Vector4f&) = default;
    Vector4f& operator=(const Vector4f&) = default;

    Vector4f(Vector4f&&) = default;
    Vector4f& operator=(Vector4f&&) = default;

    constexpr explicit Vector4f(float f) noexcept : f32{f, f, f, f} {}
    constexpr Vector4f(float x, float y, float z, float w) noexcept
        : f32{x, y, z, w}
    {
    }
    explicit Vector4f(std::span<const float, 4> span) noexcept
        : f32{span[0], span[1], span[2], span[3]}
    {
    }

    Vector4f(const Vector2f& xy, const Vector2f& zw) noexcept
        : f32{xy.x, xy.y, zw.x, zw.y}
    {
    }
    explicit Vector4f(const Vector3f& v3f) noexcept
        : f32{v3f.x, v3f.y, v3f.z, 0.0f}
    {
    }
    explicit Vector4f(const Vector3f& v3f, float w) noexcept
        : f32{v3f.x, v3f.y, v3f.z, w}
    {
    }

    inline operator Vector2f() const noexcept { return Vector2f(x, y); }
    inline operator Vector3f() const noexcept { return Vector3f(x, y, z); }

public:
    union
    {
        float f32[4];
        struct
        {
            float x, y, z, w;
        };
        struct
        {
            float r, g, b, a;
        };
    };
};

struct Matrix4x4
{
public:
    Matrix4x4() = default;

    Matrix4x4(const Matrix4x4&) = default;
    Matrix4x4& operator=(const Matrix4x4&) = default;

    Matrix4x4(Matrix4x4&&) = default;
    Matrix4x4& operator=(Matrix4x4&&) = default;

    constexpr Matrix4x4(float x0, float y0, float z0, float w0, float x1,
                        float y1, float z1, float w1, float x2, float y2,
                        float z2, float w2, float x3, float y3, float z3,
                        float w3) noexcept
        : v{Vector4f(x0, y0, z0, w0), Vector4f(x1, y1, z1, w1),
            Vector4f(x2, y2, z2, w2), Vector4f(x3, y3, z3, w3)}
    {
    }
    explicit Matrix4x4(std::span<const float, 16> span) noexcept
        : v{Vector4f(span[0], span[1], span[2], span[3]),
            Vector4f(span[4], span[5], span[6], span[7]),
            Vector4f(span[8], span[9], span[10], span[11]),
            Vector4f(span[12], span[13], span[14], span[15])}
    {
    }
    explicit Matrix4x4(const Vector4f& v0, const Vector4f& v1,
                       const Vector4f& v2, const Vector4f& v3) noexcept
        : v{v0, v1, v2, v3}
    {
    }

public:
    union
    {
        Vector4f v[4];
        struct
        {
            float m00, m01, m02, m03;
            float m10, m11, m12, m13;
            float m20, m21, m22, m23;
            float m30, m31, m32, m33;
        };
        float m[4][4];
    };
};

struct alignas(16) FMatrix
{
public:
    FMatrix() = default;

    FMatrix(const FMatrix&) = default;
    FMatrix& operator=(const FMatrix&) = default;

    FMatrix(FMatrix&&) = default;
    FMatrix& operator=(FMatrix&&) = default;

    FMatrix(float x0, float y0, float z0, float w0, float x1, float y1,
            float z1, float w1, float x2, float y2, float z2, float w2,
            float x3, float y3, float z3, float w3) noexcept;
    explicit FMatrix(std::span<const float, 16> span) noexcept;
    constexpr FMatrix(FVector v0, FVector v1, FVector v2, FVector v3) noexcept
        : v{v0, v1, v2, v3}
    {
    }

public:
    FVector v[4];
};

#ifdef _MSC_VER
    #pragma warning(pop)
#endif

// Vector2f operation

inline Vector2f operator+(const Vector2f& lhs) noexcept
{
    return lhs;
}

inline Vector2f operator-(const Vector2f& lhs) noexcept
{
    return Vector2f(-lhs.x, -lhs.y);
}

inline Vector2f& operator+=(Vector2f& lhs, const Vector2f& rhs) noexcept
{
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    return lhs;
}

inline Vector2f& operator-=(Vector2f& lhs, const Vector2f& rhs) noexcept
{
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    return lhs;
}

inline Vector2f operator*=(Vector2f& lhs, const float rhs) noexcept
{
    lhs.x *= rhs;
    lhs.y *= rhs;
    return lhs;
}

inline Vector2f operator/=(Vector2f& lhs, const float rhs)
{
    assert(rhs != 0.0f);
    lhs.x /= rhs;
    lhs.y /= rhs;
    return lhs;
}

inline Vector2f operator+(const Vector2f& lhs, const Vector2f& rhs) noexcept
{
    return Vector2f(lhs.x + rhs.x, lhs.y + rhs.y);
}

inline Vector2f operator-(const Vector2f& lhs, const Vector2f& rhs) noexcept
{
    return Vector2f(lhs.x - rhs.x, lhs.y - rhs.y);
}

inline Vector2f operator*(const Vector2f& lhs, const float rhs) noexcept
{
    return Vector2f(lhs.x * rhs, lhs.y * rhs);
}

inline Vector2f operator*(const float lhs, const Vector2f& rhs) noexcept
{
    return Vector2f(rhs.x * lhs, rhs.y * lhs);
}

inline Vector2f operator/(const Vector2f lhs, const float rhs)
{
    assert(rhs != 0.0f);
    return Vector2f(lhs.x / rhs, lhs.y / rhs);
}

inline bool operator==(const Vector2f& lhs, const Vector2f& rhs) noexcept
{
    return std::equal(lhs.f32, lhs.f32 + 2, rhs.f32);
}

inline auto operator<=>(const Vector2f& lhs, const Vector2f& rhs) noexcept
{
    return std::lexicographical_compare_three_way(lhs.f32, lhs.f32 + 2, rhs.f32,
                                                  rhs.f32 + 2);
}

inline Vector2f Abs(const Vector2f& lhs) noexcept
{
    return Vector2f(std::abs(lhs.x), std::abs(lhs.y));
}

inline float Cross(const Vector2f& lhs, const Vector2f& rhs) noexcept
{
    return lhs.x * rhs.y - lhs.x * rhs.y;
}

inline float FVectorDot(const Vector2f& lhs, const Vector2f& rhs) noexcept
{
    return lhs.x * rhs.x + lhs.y * rhs.y;
}

inline float Length2(const Vector2f& lhs) noexcept
{
    return FVectorDot(lhs, lhs);
}

inline float Length(const Vector2f& lhs) noexcept
{
    return std::sqrtf(Length2(lhs));
}

// Vector2f operation

// Vector3f operation

inline Vector3f operator+(const Vector3f& lhs) noexcept
{
    return lhs;
}

inline Vector3f operator-(const Vector3f& lhs) noexcept
{
    return Vector3f(-lhs.x, -lhs.y, -lhs.z);
}

inline Vector3f& operator+=(Vector3f& lhs, const Vector3f& rhs) noexcept
{
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
    return lhs;
}

inline Vector3f& operator-=(Vector3f& lhs, const Vector3f& rhs) noexcept
{
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    lhs.z -= rhs.z;
    return lhs;
}

inline Vector3f operator*=(Vector3f& lhs, const float rhs) noexcept
{
    lhs.x *= rhs;
    lhs.y *= rhs;
    lhs.z *= rhs;
    return lhs;
}

inline Vector3f operator/=(Vector3f& lhs, const float rhs)
{
    assert(rhs != 0.0f);
    lhs.x /= rhs;
    lhs.y /= rhs;
    lhs.z /= rhs;
    return lhs;
}

inline Vector3f operator+(const Vector3f& lhs, const Vector3f& rhs) noexcept
{
    return Vector3f(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z);
}

inline Vector3f operator-(const Vector3f& lhs, const Vector3f& rhs) noexcept
{
    return Vector3f(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z);
}

inline Vector3f operator*(const Vector3f& lhs, const float rhs) noexcept
{
    return Vector3f(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs);
}

inline Vector3f operator*(const float lhs, const Vector3f& rhs) noexcept
{
    return Vector3f(rhs.x * lhs, rhs.y * lhs, rhs.z * lhs);
}

inline Vector3f operator/(const Vector3f lhs, const float rhs)
{
    assert(rhs != 0.0f);
    return Vector3f(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs);
}

inline bool operator==(const Vector3f& lhs, const Vector3f& rhs) noexcept
{
    return std::equal(lhs.f32, lhs.f32 + 3, rhs.f32);
}

inline auto operator<=>(const Vector3f& lhs, const Vector3f& rhs) noexcept
{
    return std::lexicographical_compare_three_way(lhs.f32, lhs.f32 + 3, rhs.f32,
                                                  rhs.f32 + 3);
}

inline Vector3f Abs(const Vector3f& lhs) noexcept
{
    return Vector3f(std::abs(lhs.x), std::abs(lhs.y), std::abs(lhs.z));
}

inline Vector3f Cross(const Vector3f& lhs, const Vector3f& rhs) noexcept
{
    return Vector3f((lhs.y * rhs.z - lhs.z * rhs.y),
                    (lhs.z * rhs.x - lhs.x * rhs.z),
                    (lhs.x * rhs.y - lhs.y * rhs.x));
}

inline float FVectorDot(const Vector3f& lhs, const Vector3f& rhs) noexcept
{
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z;
}

inline float Length2(const Vector3f& lhs) noexcept
{
    return FVectorDot(lhs, lhs);
}

inline float Length(const Vector3f& lhs) noexcept
{
    return std::sqrtf(Length2(lhs));
}

// Vector3f operation

// Vector4f operation

inline Vector4f operator+(const Vector4f& lhs) noexcept
{
    return lhs;
}

inline Vector4f operator-(const Vector4f& lhs) noexcept
{
    return Vector4f(-lhs.x, -lhs.y, -lhs.z, -lhs.w);
}

inline Vector4f& operator+=(Vector4f& lhs, const Vector4f& rhs) noexcept
{
    lhs.x += rhs.x;
    lhs.y += rhs.y;
    lhs.z += rhs.z;
    lhs.w += rhs.w;
    return lhs;
}

inline Vector4f& operator-=(Vector4f& lhs, const Vector4f& rhs) noexcept
{
    lhs.x -= rhs.x;
    lhs.y -= rhs.y;
    lhs.z -= rhs.z;
    lhs.w -= rhs.w;
    return lhs;
}

inline Vector4f operator*=(Vector4f& lhs, const float rhs) noexcept
{
    lhs.x *= rhs;
    lhs.y *= rhs;
    lhs.z *= rhs;
    lhs.w *= rhs;
    return lhs;
}

inline Vector4f operator/=(Vector4f& lhs, const float rhs)
{
    assert(rhs != 0.0f);
    lhs.x /= rhs;
    lhs.y /= rhs;
    lhs.z /= rhs;
    lhs.w /= rhs;
    return lhs;
}

inline Vector4f operator+(const Vector4f& lhs, const Vector4f& rhs) noexcept
{
    return Vector4f(lhs.x + rhs.x, lhs.y + rhs.y, lhs.z + rhs.z, lhs.w + rhs.w);
}

inline Vector4f operator-(const Vector4f& lhs, const Vector4f& rhs) noexcept
{
    return Vector4f(lhs.x - rhs.x, lhs.y - rhs.y, lhs.z - rhs.z, lhs.w - rhs.w);
}

inline Vector4f operator*(const Vector4f& lhs, const float rhs) noexcept
{
    return Vector4f(lhs.x * rhs, lhs.y * rhs, lhs.z * rhs, lhs.w * rhs);
}

inline Vector4f operator*(const float lhs, const Vector4f& rhs) noexcept
{
    return Vector4f(rhs.x * lhs, rhs.y * lhs, rhs.z * lhs, rhs.w * lhs);
}

inline Vector4f operator/(const Vector4f lhs, const float rhs)
{
    assert(rhs != 0.0f);
    return Vector4f(lhs.x / rhs, lhs.y / rhs, lhs.z / rhs, lhs.w / rhs);
}

inline bool operator==(const Vector4f& lhs, const Vector4f& rhs) noexcept
{
    return std::equal(lhs.f32, lhs.f32 + 4, rhs.f32);
}

inline auto operator<=>(const Vector4f& lhs, const Vector4f& rhs) noexcept
{
    return std::lexicographical_compare_three_way(lhs.f32, lhs.f32 + 4, rhs.f32,
                                                  rhs.f32 + 4);
}

inline Vector4f Abs(const Vector4f& lhs) noexcept
{
    return Vector4f(std::abs(lhs.x), std::abs(lhs.y), std::abs(lhs.z),
                    std::abs(lhs.w));
}

inline float FVectorDot(const Vector4f& lhs, const Vector4f& rhs) noexcept
{
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z + lhs.w * rhs.w;
}

inline float Length2(const Vector4f& lhs) noexcept
{
    return FVectorDot(lhs, lhs);
}

inline float Length(const Vector4f& lhs) noexcept
{
    return std::sqrtf(Length2(lhs));
}

// Vector3f operation

// FVector operation

constexpr uint32_t ShuffleMask(uint32_t a0, uint32_t a1, uint32_t b2,
                               uint32_t b3)
{
    assert((a0 < 4) && (a1 < 4) && (b2 < 4) && (b3 < 4));
    uint32_t bits = 0u;
    bits |= a0 << 0;
    bits |= a1 << 2;
    bits |= b2 << 4;
    bits |= b3 << 6;
    return bits;
}

inline FVector LoadFVector(std::span<const float, 4> span)
{
    return _mm_loadu_ps(&span[0]);
}

inline FVector LoadFVector(const Vector4f* v4f) noexcept
{
    assert(v4f);
    return _mm_loadu_ps(&v4f->x);
}

inline void StoreFVector(Vector4f* v4f, FVector v) noexcept
{
    assert(v4f);
    _mm_storeu_ps(&v4f->x, v);
}

inline FVector SetFVector(float f0) noexcept
{
    return _mm_set_ps1(f0);
}

inline FVector SetFVector(float f0, float f1, float f2, float f3) noexcept
{
    return _mm_set_ps(f0, f1, f2, f3);
}

inline FVector FVectorSwizzle(FVector v, uint32_t E0, uint32_t E1, uint32_t E2,
                              uint32_t E3) noexcept
{
    assert((E0 < 4) && (E1 < 4) && (E2 < 4) && (E3 < 4));
    uint32_t pos[4] = {E0, E1, E2, E3};
    __m128i vi = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&pos[0]));
    return _mm_permutevar_ps(v, vi);
}

inline FVector FVectorShuffle(FVector v0, FVector v1, uint32_t mask)
{
    return _mm_shuffle_ps(v0, v1, mask);
}

inline FVector FVectorNegate(FVector v0) noexcept
{
    FVector sign = _mm_set1_ps(-0.0f);
    return _mm_xor_ps(v0, sign);
}

inline FVector FVectorAdd(FVector v0, FVector v1) noexcept
{
    return _mm_add_ps(v0, v1);
}

inline FVector FVectorSubtract(FVector v0, FVector v1) noexcept
{
    return _mm_sub_ps(v0, v1);
}

inline FVector FVectorMAdd(FVector v0, FVector v1, FVector v2) noexcept
{
    return _mm_fmadd_ps(v0, v1, v2);
}

inline FVector FVectorMultiply(FVector v0, FVector v1) noexcept
{
    return _mm_mul_ps(v0, v1);
}

inline FVector FVectorMultiply(FVector v0, float f) noexcept
{
    FVector v1 = _mm_set1_ps(f);
    return _mm_mul_ps(v0, v1);
}

inline FVector FVectorMultiply(float f, FVector v1) noexcept
{
    FVector v0 = _mm_set1_ps(f);
    return _mm_mul_ps(v0, v1);
}

inline FVector FVectorDivide(FVector v0, FVector v1) noexcept
{
    return _mm_div_ps(v0, v1);
}

inline FVector FVectorDivide(FVector v0, float f) noexcept
{
    assert(f != 0.0f);
    FVector v1 = _mm_set1_ps(f);
    return _mm_div_ps(v0, v1);
}

inline FVector FVectorReciprocal(FVector v0) noexcept
{
    return _mm_rcp_ps(v0);
}

inline FVector FVectorDot(FVector v0, FVector v1) noexcept
{
    FVector xyzw = _mm_mul_ps(v0, v1);
    FVector yxwz = FVectorSwizzle(xyzw, 1, 0, 3, 2);
    xyzw = _mm_add_ps(xyzw, yxwz);
    yxwz = FVectorSwizzle(xyzw, 2, 3, 0, 1);
    return _mm_add_ps(xyzw, yxwz);
}

// FVector operation

// Matrix4x4 operation

inline Matrix4x4 operator+(const Matrix4x4& lhs) noexcept
{
    return lhs;
}

inline Matrix4x4 operator-(const Matrix4x4& rhs) noexcept
{
    return Matrix4x4(-rhs.v[0], -rhs.v[1], -rhs.v[2], -rhs.v[3]);
}

// Matrix4x4 operation

// FMatrix operation

inline FMatrix::FMatrix(float x0, float y0, float z0, float w0, float x1,
                        float y1, float z1, float w1, float x2, float y2,
                        float z2, float w2, float x3, float y3, float z3,
                        float w3) noexcept
{
    v[0] = SetFVector(x0, y0, z0, w0);
    v[1] = SetFVector(x1, y1, z1, w1);
    v[2] = SetFVector(x2, y2, z2, w2);
    v[3] = SetFVector(x3, y3, z3, w3);
}

inline FMatrix::FMatrix(std::span<const float, 16> span) noexcept
{
    v[0] = LoadFVector(span.subspan<0, 4>());
    v[1] = LoadFVector(span.subspan<4, 4>());
    v[2] = LoadFVector(span.subspan<8, 4>());
    v[3] = LoadFVector(span.subspan<12, 4>());
}

inline FMatrix FMatrixTranspose(const FMatrix& m0) noexcept
{
    FMatrix tr;
    return tr;
}

inline FMatrix operator-(const FMatrix& lhs) noexcept
{
    FMatrix m;
    m.v[0] = FVectorNegate(lhs.v[0]);
    m.v[1] = FVectorNegate(lhs.v[1]);
    m.v[2] = FVectorNegate(lhs.v[2]);
    m.v[3] = FVectorNegate(lhs.v[3]);
    return m;
}

inline FMatrix operator+(const FMatrix& lhs, const FMatrix& rhs) noexcept
{
    FMatrix m;
    m.v[0] = FVectorAdd(lhs.v[0], rhs.v[0]);
    m.v[1] = FVectorAdd(lhs.v[1], rhs.v[1]);
    m.v[2] = FVectorAdd(lhs.v[2], rhs.v[2]);
    m.v[3] = FVectorAdd(lhs.v[3], rhs.v[3]);
    return m;
}

inline FMatrix operator-(const FMatrix& lhs, const FMatrix& rhs) noexcept
{
    FMatrix m;
    m.v[0] = FVectorSubtract(lhs.v[0], rhs.v[0]);
    m.v[1] = FVectorSubtract(lhs.v[1], rhs.v[1]);
    m.v[2] = FVectorSubtract(lhs.v[2], rhs.v[2]);
    m.v[3] = FVectorSubtract(lhs.v[3], rhs.v[3]);
    return m;
}

inline FVector operator*(FVector lhs, const FMatrix& rhs) noexcept
{
    FVector v;
    v = FVectorMultiply(FVectorSwizzle(lhs, 0, 0, 0, 0), rhs.v[0]);
    v = FVectorMAdd(FVectorSwizzle(lhs, 1, 1, 1, 1), rhs.v[1], v);
    v = FVectorMAdd(FVectorSwizzle(lhs, 2, 2, 2, 2), rhs.v[2], v);
    v = FVectorMAdd(FVectorSwizzle(lhs, 3, 3, 3, 3), rhs.v[3], v);
    return v;
}

inline FMatrix operator*(const FMatrix& lhs, const FMatrix& rhs) noexcept
{
    FVector v0 = FVectorShuffle(rhs.v[0], rhs.v[1], ShuffleMask(0, 1, 0, 1));
    FVector v1 = FVectorShuffle(rhs.v[2], rhs.v[3], ShuffleMask(0, 1, 0, 1));
    FVector v2 = FVectorShuffle(rhs.v[0], rhs.v[1], ShuffleMask(2, 3, 2, 3));
    FVector v3 = FVectorShuffle(rhs.v[2], rhs.v[3], ShuffleMask(2, 3, 2, 3));

    FMatrix m;
    m.v[0] = FVectorShuffle(v0, v1, ShuffleMask(0, 2, 0, 2));
    m.v[1] = FVectorShuffle(v0, v1, ShuffleMask(1, 3, 1, 3));
    m.v[2] = FVectorShuffle(v2, v3, ShuffleMask(0, 2, 0, 2));
    m.v[3] = FVectorShuffle(v2, v3, ShuffleMask(1, 3, 1, 3));
    return m;
}

// FMatrix operation

}  // namespace leo

#endif  // LEOMATH_VECTOR_H_