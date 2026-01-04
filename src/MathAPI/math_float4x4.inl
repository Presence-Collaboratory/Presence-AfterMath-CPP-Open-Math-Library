#ifndef MATH_FLOAT4X4_INL
#define MATH_FLOAT4X4_INL

#include "math_float4x4.h"
#include "math_float3x3.h"
#include "math_quaternion.h"

namespace Math
{
    // ============================================================================
    // Constructors
    // ============================================================================

    /**
     * @brief Default constructor (initializes to identity matrix)
     */
    inline float4x4::float4x4() noexcept
        : col0_(1, 0, 0, 0), col1_(0, 1, 0, 0), col2_(0, 0, 1, 0), col3_(0, 0, 0, 1) {}

    /**
     * @brief Construct from column vectors
     * @param c0 First column vector
     * @param c1 Second column vector
     * @param c2 Third column vector
     * @param c3 Fourth column vector (translation/perspective)
     */
    inline float4x4::float4x4(const float4& c0, const float4& c1, const float4& c2, const float4& c3) noexcept
        : col0_(c0), col1_(c1), col2_(c2), col3_(c3) {}

    /**
     * @brief Construct from 16 scalar values (column-major order)
     */
    inline float4x4::float4x4(float m00, float m10, float m20, float m30,
        float m01, float m11, float m21, float m31,
        float m02, float m12, float m22, float m32,
        float m03, float m13, float m23, float m33) noexcept
        : col0_(m00, m10, m20, m30)
        , col1_(m01, m11, m21, m31)
        , col2_(m02, m12, m22, m32)
        , col3_(m03, m13, m23, m33) {}

    /**
     * @brief Construct from column-major array
     * @param data Column-major array of 16 elements
     */
    inline float4x4::float4x4(const float* data) noexcept
        : col0_(data[0], data[1], data[2], data[3])
        , col1_(data[4], data[5], data[6], data[7])
        , col2_(data[8], data[9], data[10], data[11])
        , col3_(data[12], data[13], data[14], data[15]) {}

    /**
     * @brief Construct from scalar (diagonal matrix)
     */
    inline float4x4::float4x4(float scalar) noexcept
        : col0_(scalar, 0, 0, 0), col1_(0, scalar, 0, 0), col2_(0, 0, scalar, 0), col3_(0, 0, 0, scalar) {}

    /**
     * @brief Construct from diagonal vector
     */
    inline float4x4::float4x4(const float4& diagonal) noexcept
        : col0_(diagonal.x, 0, 0, 0), col1_(0, diagonal.y, 0, 0), col2_(0, 0, diagonal.z, 0), col3_(0, 0, 0, diagonal.w) {}

    /**
     * @brief Construct from 3x3 matrix (extends to 4x4 with identity)
     * @note float3x3 is assumed to be row-major, need to transpose for column-major
     */
    inline float4x4::float4x4(const float3x3& m) noexcept {
        // Extract rows from 3x3 matrix and set as columns for column-major
        float3 r0 = m.row0();
        float3 r1 = m.row1();
        float3 r2 = m.row2();

        col0_ = float4(r0.x, r1.x, r2.x, 0.0f);
        col1_ = float4(r0.y, r1.y, r2.y, 0.0f);
        col2_ = float4(r0.z, r1.z, r2.z, 0.0f);
        col3_ = float4(0.0f, 0.0f, 0.0f, 1.0f);
    }

    /**
     * @brief Construct from quaternion (rotation matrix)
     * @note For column-major, we need to set the quaternion rotation matrix in columns
     */
    inline float4x4::float4x4(const quaternion& q) noexcept {
        float x = q.x, y = q.y, z = q.z, w = q.w;

        // Compute common values
        float xx = x * x;
        float yy = y * y;
        float zz = z * z;
        float xy = x * y;
        float xz = x * z;
        float yz = y * z;
        float wx = w * x;
        float wy = w * y;
        float wz = w * z;

        // Column 0: [1 - 2*(yy + zz), 2*(xy + wz), 2*(xz - wy), 0]
        float m00 = 1.0f - 2.0f * (yy + zz);
        float m10 = 2.0f * (xy + wz);
        float m20 = 2.0f * (xz - wy);
        float m30 = 0.0f;

        // Column 1: [2*(xy - wz), 1 - 2*(xx + zz), 2*(yz + wx), 0]
        float m01 = 2.0f * (xy - wz);
        float m11 = 1.0f - 2.0f * (xx + zz);
        float m21 = 2.0f * (yz + wx);
        float m31 = 0.0f;

        // Column 2: [2*(xz + wy), 2*(yz - wx), 1 - 2*(xx + yy), 0]
        float m02 = 2.0f * (xz + wy);
        float m12 = 2.0f * (yz - wx);
        float m22 = 1.0f - 2.0f * (xx + yy);
        float m32 = 0.0f;

        // Column 3: [0, 0, 0, 1]
        float m03 = 0.0f;
        float m13 = 0.0f;
        float m23 = 0.0f;
        float m33 = 1.0f;

        // Set columns
        col0_ = float4(m00, m10, m20, m30);
        col1_ = float4(m01, m11, m21, m31);
        col2_ = float4(m02, m12, m22, m32);
        col3_ = float4(m03, m13, m23, m33);
    }

    // ============================================================================
    // Static Constructors
    // ============================================================================

    inline float4x4 float4x4::identity() noexcept { return float4x4(); }
    inline float4x4 float4x4::zero() noexcept { return float4x4(0.0f); }

    // --- Transformations ---

    inline float4x4 float4x4::translation(float x, float y, float z) noexcept {
        return float4x4(1, 0, 0, 0,
            0, 1, 0, 0,
            0, 0, 1, 0,
            x, y, z, 1);
    }

    inline float4x4 float4x4::translation(const float3& p) noexcept {
        return translation(p.x, p.y, p.z);
    }

    inline float4x4 float4x4::scaling(float x, float y, float z) noexcept {
        return float4x4(x, 0, 0, 0,
            0, y, 0, 0,
            0, 0, z, 0,
            0, 0, 0, 1);
    }

    inline float4x4 float4x4::scaling(const float3& s) noexcept {
        return scaling(s.x, s.y, s.z);
    }

    inline float4x4 float4x4::scaling(float s) noexcept {
        return scaling(s, s, s);
    }

    inline float4x4 float4x4::rotation_x(float angle) noexcept {
        float s, c;
        MathFunctions::sin_cos(angle, &s, &c);
        return float4x4(1, 0, 0, 0,
            0, c, s, 0,
            0, -s, c, 0,
            0, 0, 0, 1);
    }

    inline float4x4 float4x4::rotation_y(float angle) noexcept {
        float s, c;
        MathFunctions::sin_cos(angle, &s, &c);
        return float4x4(c, 0, -s, 0,
            0, 1, 0, 0,
            s, 0, c, 0,
            0, 0, 0, 1);
    }

    inline float4x4 float4x4::rotation_z(float angle) noexcept {
        float s, c;
        MathFunctions::sin_cos(angle, &s, &c);
        return float4x4(c, s, 0, 0,
            -s, c, 0, 0,
            0, 0, 1, 0,
            0, 0, 0, 1);
    }

    inline float4x4 float4x4::rotation_axis(const float3& axis, float angle) noexcept {
        float s, c;
        MathFunctions::sin_cos(angle, &s, &c);
        float t = 1.0f - c;
        float x = axis.x, y = axis.y, z = axis.z;
        return float4x4(
            t * x * x + c, t * x * y - z * s, t * x * z + y * s, 0,
            t * x * y + z * s, t * y * y + c, t * y * z - x * s, 0,
            t * x * z - y * s, t * y * z + x * s, t * z * z + c, 0,
            0, 0, 0, 1
        );
    }

    inline float4x4 float4x4::rotation_euler(const float3& a) noexcept {
        return rotation_z(a.z) * rotation_y(a.y) * rotation_x(a.x);
    }

    inline float4x4 float4x4::TRS(const float3& t, const quaternion& r, const float3& s) noexcept {
        return translation(t) * float4x4(r) * scaling(s);
    }

    // --- Projections ---

    inline float4x4 float4x4::perspective_lh_zo(float fov, float ar, float zn, float zf) noexcept {
        float h = 1.0f / std::tan(fov * 0.5f);
        float w = h / ar;
        float r = zf / (zf - zn);
        return float4x4(w, 0, 0, 0,
            0, h, 0, 0,
            0, 0, r, 1,
            0, 0, -r * zn, 0);
    }

    inline float4x4 float4x4::perspective_rh_zo(float fov, float ar, float zn, float zf) noexcept {
        float h = 1.0f / std::tan(fov * 0.5f);
        float w = h / ar;
        float r = zf / (zn - zf);
        return float4x4(w, 0, 0, 0,
            0, h, 0, 0,
            0, 0, r, -1,
            0, 0, r * zn, 0);
    }

    inline float4x4 float4x4::perspective_lh_no(float fov, float ar, float zn, float zf) noexcept {
        float h = 1.0f / std::tan(fov * 0.5f);
        float w = h / ar;
        return float4x4(w, 0, 0, 0,
            0, h, 0, 0,
            0, 0, (zf + zn) / (zf - zn), 1,
            0, 0, -2 * zn * zf / (zf - zn), 0);
    }

    inline float4x4 float4x4::perspective_rh_no(float fov, float ar, float zn, float zf) noexcept {
        float h = 1.0f / std::tan(fov * 0.5f);
        float w = h / ar;
        return float4x4(w, 0, 0, 0,
            0, h, 0, 0,
            0, 0, -(zf + zn) / (zf - zn), -1,
            0, 0, -2 * zn * zf / (zf - zn), 0);
    }

    inline float4x4 float4x4::perspective(float fov, float ar, float zn, float zf) noexcept {
        return perspective_rh_zo(fov, ar, zn, zf);
    }

    inline float4x4 float4x4::orthographic_lh_zo(float width, float height, float zNear, float zFar) noexcept {
        float fRange = 1.0f / (zFar - zNear);
        return float4x4(
            2.0f / width, 0.0f, 0.0f, 0.0f,
            0.0f, 2.0f / height, 0.0f, 0.0f,
            0.0f, 0.0f, fRange, 0.0f,
            0.0f, 0.0f, -zNear * fRange, 1.0f
        );
    }

    inline float4x4 float4x4::orthographic_off_center_lh_zo(float left, float right, float bottom, float top, float zNear, float zFar) noexcept {
        float fRange = 1.0f / (zFar - zNear);
        return float4x4(
            2.0f / (right - left), 0.0f, 0.0f, 0.0f,
            0.0f, 2.0f / (top - bottom), 0.0f, 0.0f,
            0.0f, 0.0f, fRange, 0.0f,
            -(left + right) / (right - left), -(top + bottom) / (top - bottom), -zNear * fRange, 1.0f
        );
    }

    inline float4x4 float4x4::orthographic(float w, float h, float zn, float zf) noexcept {
        return orthographic_lh_zo(w, h, zn, zf);
    }

    // --- Cameras ---

    inline float4x4 float4x4::look_at_lh(const float3& eye, const float3& target, const float3& up) noexcept {
        float3 z = (target - eye).normalize();   // Forward
        float3 x = up.cross(z).normalize();      // Right
        float3 y = z.cross(x);                   // Up

        // Translation = -dot(axis, eye)
        float tx = -x.dot(eye);
        float ty = -y.dot(eye);
        float tz = -z.dot(eye);

        // For column-major: axes become columns
        return float4x4(
            x.x, y.x, z.x, 0.0f,  // Column 0: x-axis components
            x.y, y.y, z.y, 0.0f,  // Column 1: y-axis components  
            x.z, y.z, z.z, 0.0f,  // Column 2: z-axis components
            tx, ty, tz, 1.0f   // Column 3: translation
        );
    }

    inline float4x4 float4x4::look_at_rh(const float3& eye, const float3& target, const float3& up) noexcept {
        float3 z = (eye - target).normalize();   // Backwards (RH)
        float3 x = up.cross(z).normalize();      // Right
        float3 y = z.cross(x);                   // Up

        // Translation = -dot(axis, eye)
        float tx = -x.dot(eye);
        float ty = -y.dot(eye);
        float tz = -z.dot(eye);

        return float4x4(
            x.x, y.x, z.x, 0.0f,
            x.y, y.y, z.y, 0.0f,
            x.z, y.z, z.z, 0.0f,
            tx, ty, tz, 1.0f
        );
    }

    inline float4x4 float4x4::look_at(const float3& eye, const float3& target, const float3& up) noexcept {
        return look_at_rh(eye, target, up);
    }

    // ============================================================================
    // Access Operators
    // ============================================================================

    inline float4& float4x4::operator[](int colIndex) noexcept {
        return (&col0_)[colIndex];
    }

    inline const float4& float4x4::operator[](int colIndex) const noexcept {
        return (&col0_)[colIndex];
    }

    inline float& float4x4::operator()(int row, int col) noexcept {
        return (&col0_)[col][row];
    }

    inline const float& float4x4::operator()(int row, int col) const noexcept {
        return (&col0_)[col][row];
    }

    // ============================================================================
    // Column and Row Accessors
    // ============================================================================

    inline float4 float4x4::row0() const noexcept {
        return float4(col0_.x, col1_.x, col2_.x, col3_.x);
    }

    inline float4 float4x4::row1() const noexcept {
        return float4(col0_.y, col1_.y, col2_.y, col3_.y);
    }

    inline float4 float4x4::row2() const noexcept {
        return float4(col0_.z, col1_.z, col2_.z, col3_.z);
    }

    inline float4 float4x4::row3() const noexcept {
        return float4(col0_.w, col1_.w, col2_.w, col3_.w);
    }

    // ============================================================================
    // Compound Assignment Operators
    // ============================================================================

    inline float4x4& float4x4::operator+=(const float4x4& rhs) noexcept {
        col0_ += rhs.col0_;
        col1_ += rhs.col1_;
        col2_ += rhs.col2_;
        col3_ += rhs.col3_;
        return *this;
    }

    inline float4x4& float4x4::operator-=(const float4x4& rhs) noexcept {
        col0_ -= rhs.col0_;
        col1_ -= rhs.col1_;
        col2_ -= rhs.col2_;
        col3_ -= rhs.col3_;
        return *this;
    }

    inline float4x4& float4x4::operator*=(float s) noexcept {
        col0_ *= s;
        col1_ *= s;
        col2_ *= s;
        col3_ *= s;
        return *this;
    }

    inline float4x4& float4x4::operator/=(float s) noexcept {
        float is = 1.0f / s;
        col0_ *= is;
        col1_ *= is;
        col2_ *= is;
        col3_ *= is;
        return *this;
    }

    inline float4x4& float4x4::operator*=(const float4x4& rhs) noexcept {
        *this = *this * rhs;
        return *this;
    }

    // ============================================================================
    // Unary Operators
    // ============================================================================

    inline float4x4 float4x4::operator+() const noexcept {
        return *this;
    }

    inline float4x4 float4x4::operator-() const noexcept {
        return float4x4(-col0_, -col1_, -col2_, -col3_);
    }

    // ============================================================================
    // Matrix Operations
    // ============================================================================

    /**
     * @brief Compute transposed matrix (SSE optimized)
     */
    inline float4x4 float4x4::transposed() const noexcept {
        __m128 t0 = _mm_shuffle_ps(col0_.get_simd(), col1_.get_simd(), 0x44);
        __m128 t2 = _mm_shuffle_ps(col0_.get_simd(), col1_.get_simd(), 0xEE);
        __m128 t1 = _mm_shuffle_ps(col2_.get_simd(), col3_.get_simd(), 0x44);
        __m128 t3 = _mm_shuffle_ps(col2_.get_simd(), col3_.get_simd(), 0xEE);
        return float4x4(
            float4(_mm_shuffle_ps(t0, t1, 0x88)),
            float4(_mm_shuffle_ps(t0, t1, 0xDD)),
            float4(_mm_shuffle_ps(t2, t3, 0x88)),
            float4(_mm_shuffle_ps(t2, t3, 0xDD))
        );
    }

    /**
     * @brief Compute matrix determinant
     */
    inline float float4x4::determinant() const noexcept {
        float m00 = col0_.x, m10 = col0_.y, m20 = col0_.z, m30 = col0_.w;
        float m01 = col1_.x, m11 = col1_.y, m21 = col1_.z, m31 = col1_.w;
        float m02 = col2_.x, m12 = col2_.y, m22 = col2_.z, m32 = col2_.w;
        float m03 = col3_.x, m13 = col3_.y, m23 = col3_.z, m33 = col3_.w;

        return m03 * m12 * m21 * m30 - m02 * m13 * m21 * m30 - m03 * m11 * m22 * m30 + m01 * m13 * m22 * m30 +
            m02 * m11 * m23 * m30 - m01 * m12 * m23 * m30 - m03 * m12 * m20 * m31 + m02 * m13 * m20 * m31 +
            m03 * m10 * m22 * m31 - m00 * m13 * m22 * m31 - m02 * m10 * m23 * m31 + m00 * m12 * m23 * m31 +
            m03 * m11 * m20 * m32 - m01 * m13 * m20 * m32 - m03 * m10 * m21 * m32 + m00 * m13 * m21 * m32 +
            m01 * m10 * m23 * m32 - m00 * m11 * m23 * m32 - m02 * m11 * m20 * m33 + m01 * m12 * m20 * m33 +
            m02 * m10 * m21 * m33 - m00 * m12 * m21 * m33 - m01 * m10 * m22 * m33 + m00 * m11 * m22 * m33;
    }

    inline float4x4 float4x4::inverted_affine() const noexcept {
        const float3 c0 = col0_.xyz();
        const float3 c1 = col1_.xyz();
        const float3 c2 = col2_.xyz();
        const float3 t = get_translation();

        // Cross products compute unnormalized ROWS of the inverse linear part
        const float3 r0 = c1.cross(c2);  // First row of 3x3 inverse
        const float3 r1 = c2.cross(c0);  // Second row of 3x3 inverse
        const float3 r2 = c0.cross(c1);  // Third row of 3x3 inverse

        const float det = c0.dot(r0);

        if (std::abs(det) < Constants::Constants<float>::Epsilon) {
            return identity();
        }

        const float inv_det = 1.0f / det;

        // Apply inverse determinant to get actual rows
        const float3 inv_row0 = r0 * inv_det;
        const float3 inv_row1 = r1 * inv_det;
        const float3 inv_row2 = r2 * inv_det;

        // Calculate inverse translation: T' = -InvLinear * T
        float tx = -inv_row0.dot(t);
        float ty = -inv_row1.dot(t);
        float tz = -inv_row2.dot(t);

        // For column-major: rows become columns
        // inv_row0.x becomes column0.x, inv_row1.x becomes column0.y, etc.
        return float4x4(
            inv_row0.x, inv_row1.x, inv_row2.x, 0.0f,  // Column 0
            inv_row0.y, inv_row1.y, inv_row2.y, 0.0f,  // Column 1
            inv_row0.z, inv_row1.z, inv_row2.z, 0.0f,  // Column 2
            tx, ty, tz, 1.0f                           // Column 3
        );
    }

    inline float4x4 float4x4::inverted() const noexcept {
        if (is_affine(Constants::Constants<float>::Epsilon)) {
            return inverted_affine();
        }

        const float det = determinant();
        if (std::abs(det) < Constants::Constants<float>::Epsilon) {
            return identity();
        }

        return adjugate() * (1.0f / det);
    }

    /**
     * @brief Compute adjugate matrix (Fully SSE optimized)
     */
    inline float4x4 float4x4::adjugate() const noexcept {
        float a = col0_.x, e = col0_.y, i = col0_.z, m = col0_.w;
        float b = col1_.x, f = col1_.y, j = col1_.z, n = col1_.w;
        float c = col2_.x, g = col2_.y, k = col2_.z, o = col2_.w;
        float d = col3_.x, h = col3_.y, l = col3_.z, p = col3_.w;

        // Compute all 2x2 determinants
        float kp_lo = k * p - l * o;
        float jp_ln = j * p - l * n;
        float jo_kn = j * o - k * n;
        float ip_lm = i * p - l * m;
        float io_km = i * o - k * m;
        float in_jm = i * n - j * m;

        float gp_ho = g * p - h * o;
        float fp_hn = f * p - h * n;
        float fo_gn = f * o - g * n;
        float ep_hm = e * p - h * m;
        float eo_gm = e * o - g * m;
        float en_fm = e * n - f * m;

        float gl_hk = g * l - h * k;
        float fl_hj = f * l - h * j;
        float fk_gj = f * k - g * j;
        float el_hi = e * l - h * i;
        float ek_gi = e * k - g * i;
        float ej_fi = e * j - f * i;

        // Compute adjugate matrix elements
        float m00 = (f * kp_lo - g * jp_ln + h * jo_kn);
        float m10 = -(e * kp_lo - g * ip_lm + h * io_km);
        float m20 = (e * jp_ln - f * ip_lm + h * in_jm);
        float m30 = -(e * jo_kn - f * io_km + g * in_jm);

        float m01 = -(b * kp_lo - c * jp_ln + d * jo_kn);
        float m11 = (a * kp_lo - c * ip_lm + d * io_km);
        float m21 = -(a * jp_ln - b * ip_lm + d * in_jm);
        float m31 = (a * jo_kn - b * io_km + c * in_jm);

        float m02 = (b * gp_ho - c * fp_hn + d * fo_gn);
        float m12 = -(a * gp_ho - c * ep_hm + d * eo_gm);
        float m22 = (a * fp_hn - b * ep_hm + d * en_fm);
        float m32 = -(a * fo_gn - b * eo_gm + c * en_fm);

        float m03 = -(b * gl_hk - c * fl_hj + d * fk_gj);
        float m13 = (a * gl_hk - c * el_hi + d * ek_gi);
        float m23 = -(a * fl_hj - b * el_hi + d * ej_fi);
        float m33 = (a * fk_gj - b * ek_gi + c * ej_fi);

        return float4x4(
            m00, m10, m20, m30,
            m01, m11, m21, m31,
            m02, m12, m22, m32,
            m03, m13, m23, m33
        );
    }

    inline float3x3 float4x4::normal_matrix() const noexcept {
        float3x3 mat3x3(
            float3(col0_.x, col0_.y, col0_.z),
            float3(col1_.x, col1_.y, col1_.z),
            float3(col2_.x, col2_.y, col2_.z)
        );
        return mat3x3.inverted().transposed();
    }

    inline float float4x4::trace() const noexcept {
        return col0_.x + col1_.y + col2_.z + col3_.w;
    }

    inline float4 float4x4::diagonal() const noexcept {
        return float4(col0_.x, col1_.y, col2_.z, col3_.w);
    }

    inline float float4x4::frobenius_norm() const noexcept {
        return std::sqrt(col0_.length_sq() + col1_.length_sq() + col2_.length_sq() + col3_.length_sq());
    }

    // ============================================================================
    // Vector Transformations
    // ============================================================================

    inline float4 float4x4::transform_vector(const float4& v) const noexcept {
        __m128 r = _mm_mul_ps(col0_.get_simd(), _mm_set1_ps(v.x));
        r = _mm_add_ps(r, _mm_mul_ps(col1_.get_simd(), _mm_set1_ps(v.y)));
        r = _mm_add_ps(r, _mm_mul_ps(col2_.get_simd(), _mm_set1_ps(v.z)));
        r = _mm_add_ps(r, _mm_mul_ps(col3_.get_simd(), _mm_set1_ps(v.w)));
        return float4(r);
    }

    inline float3 float4x4::transform_point(const float3& p) const noexcept {
        float4 r = transform_vector(float4(p, 1.0f));
        return float3(r.x, r.y, r.z) / r.w;
    }

    inline float3 float4x4::transform_vector(const float3& v) const noexcept {
        float4 r = transform_vector(float4(v, 0.0f));
        return float3(r.x, r.y, r.z);
    }

    inline float3 float4x4::transform_direction(const float3& d) const noexcept {
        float4 r = transform_vector(float4(d, 0.0f));
        return float3(r.x, r.y, r.z).normalize();
    }

    // ============================================================================
    // Transformation Component Extraction
    // ============================================================================

    inline float3 float4x4::get_translation() const noexcept {
        return float3(col3_.x, col3_.y, col3_.z);
    }

    inline float3 float4x4::get_scale() const noexcept {
        return float3(
            float3(col0_.x, col0_.y, col0_.z).length(),
            float3(col1_.x, col1_.y, col1_.z).length(),
            float3(col2_.x, col2_.y, col2_.z).length()
        );
    }

    inline quaternion float4x4::get_rotation() const noexcept {
        float3 col0_norm = float3(col0_.x, col0_.y, col0_.z).normalize();
        float3 col1_norm = float3(col1_.x, col1_.y, col1_.z).normalize();
        float3 col2_norm = float3(col2_.x, col2_.y, col2_.z).normalize();

        float3x3 rot_matrix(col0_norm, col1_norm, col2_norm);
        return quaternion::from_matrix(rot_matrix);
    }

    inline void float4x4::set_translation(const float3& t) noexcept {
        col3_.x = t.x;
        col3_.y = t.y;
        col3_.z = t.z;
    }

    inline void float4x4::set_scale(const float3& s) noexcept {
        const float3 current_scale = get_scale();
        const float eps = 1e-8f;

        // Handle each axis separately
        float3 c0 = float3(col0_.x, col0_.y, col0_.z);
        float3 c1 = float3(col1_.x, col1_.y, col1_.z);
        float3 c2 = float3(col2_.x, col2_.y, col2_.z);

        if (c0.length_sq() > eps) {
            c0 = c0.normalize() * s.x;
        }
        else {
            c0 = float3(s.x, 0.0f, 0.0f);
        }

        if (c1.length_sq() > eps) {
            c1 = c1.normalize() * s.y;
        }
        else {
            c1 = float3(0.0f, s.y, 0.0f);
        }

        if (c2.length_sq() > eps) {
            c2 = c2.normalize() * s.z;
        }
        else {
            c2 = float3(0.0f, 0.0f, s.z);
        }

        col0_.x = c0.x; col0_.y = c0.y; col0_.z = c0.z;
        col1_.x = c1.x; col1_.y = c1.y; col1_.z = c1.z;
        col2_.x = c2.x; col2_.y = c2.y; col2_.z = c2.z;
    }

    // ============================================================================
    // Utility Methods
    // ============================================================================

    inline bool float4x4::is_identity(float epsilon) const noexcept {
        return col0_.approximately(float4(1, 0, 0, 0), epsilon) &&
            col1_.approximately(float4(0, 1, 0, 0), epsilon) &&
            col2_.approximately(float4(0, 0, 1, 0), epsilon) &&
            col3_.approximately(float4(0, 0, 0, 1), epsilon);
    }

    inline bool float4x4::is_affine(float eps) const noexcept {
        return std::abs(col3_.w - 1.0f) < eps &&
            col0_.w < eps&&
            col1_.w < eps&&
            col2_.w < eps;
    }

    inline bool float4x4::is_orthogonal(float epsilon) const noexcept {
        if (!is_affine(epsilon)) return false;

        const float3 col0_xyz = col0_.xyz();
        const float3 col1_xyz = col1_.xyz();
        const float3 col2_xyz = col2_.xyz();

        float dot01 = std::abs(col0_xyz.dot(col1_xyz));
        float dot02 = std::abs(col0_xyz.dot(col2_xyz));
        float dot12 = std::abs(col1_xyz.dot(col2_xyz));

        if (dot01 > epsilon || dot02 > epsilon || dot12 > epsilon) {
            return false;
        }

        float len0 = col0_xyz.length_sq();
        float len1 = col1_xyz.length_sq();
        float len2 = col2_xyz.length_sq();

        return MathFunctions::approximately(len0, 1.0f, epsilon) &&
            MathFunctions::approximately(len1, 1.0f, epsilon) &&
            MathFunctions::approximately(len2, 1.0f, epsilon);
    }

    inline bool float4x4::approximately(const float4x4& o, float e) const noexcept {
        return col0_.approximately(o.col0_, e) &&
            col1_.approximately(o.col1_, e) &&
            col2_.approximately(o.col2_, e) &&
            col3_.approximately(o.col3_, e);
    }

    inline bool float4x4::approximately_zero(float e) const noexcept {
        return approximately(zero(), e);
    }

    inline std::string float4x4::to_string() const {
        char buf[512];
        snprintf(buf, sizeof(buf),
            "[%f %f %f %f]\n"
            "[%f %f %f %f]\n"
            "[%f %f %f %f]\n"
            "[%f %f %f %f]",
            col0_.x, col1_.x, col2_.x, col3_.x,
            col0_.y, col1_.y, col2_.y, col3_.y,
            col0_.z, col1_.z, col2_.z, col3_.z,
            col0_.w, col1_.w, col2_.w, col3_.w);
        return std::string(buf);
    }

    inline void float4x4::to_column_major(float* data) const noexcept {
        data[0] = col0_.x; data[1] = col0_.y; data[2] = col0_.z; data[3] = col0_.w;
        data[4] = col1_.x; data[5] = col1_.y; data[6] = col1_.z; data[7] = col1_.w;
        data[8] = col2_.x; data[9] = col2_.y; data[10] = col2_.z; data[11] = col2_.w;
        data[12] = col3_.x; data[13] = col3_.y; data[14] = col3_.z; data[15] = col3_.w;
    }

    inline void float4x4::to_row_major(float* data) const noexcept {
        data[0] = col0_.x; data[1] = col1_.x; data[2] = col2_.x; data[3] = col3_.x;
        data[4] = col0_.y; data[5] = col1_.y; data[6] = col2_.y; data[7] = col3_.y;
        data[8] = col0_.z; data[9] = col1_.z; data[10] = col2_.z; data[11] = col3_.z;
        data[12] = col0_.w; data[13] = col1_.w; data[14] = col2_.w; data[15] = col3_.w;
    }

    // ============================================================================
    // Comparison Operators
    // ============================================================================

    inline bool float4x4::operator==(const float4x4& rhs) const noexcept {
        return approximately(rhs);
    }

    inline bool float4x4::operator!=(const float4x4& rhs) const noexcept {
        return !(*this == rhs);
    }

    // ============================================================================
    // Binary Operators
    // ============================================================================

    /**
     * @brief Matrix multiplication (SSE optimized for column-major)
     */
    inline float4x4 operator*(const float4x4& lhs, const float4x4& rhs) noexcept {
        float4x4 res;

        // Direct access to columns
        const float4* rhs_cols = &rhs.col0_;
        float4* res_cols = &res.col0_;

        // For each column in result
        for (int i = 0; i < 4; ++i) {
            // Load current column from rhs
            __m128 col = rhs_cols[i].get_simd();

            // Broadcast components
            __m128 x = _mm_shuffle_ps(col, col, _MM_SHUFFLE(0, 0, 0, 0));
            __m128 y = _mm_shuffle_ps(col, col, _MM_SHUFFLE(1, 1, 1, 1));
            __m128 z = _mm_shuffle_ps(col, col, _MM_SHUFFLE(2, 2, 2, 2));
            __m128 w = _mm_shuffle_ps(col, col, _MM_SHUFFLE(3, 3, 3, 3));

            // Linear combination of LHS columns
            __m128 r = _mm_mul_ps(lhs.col0_.get_simd(), x);
            r = _mm_add_ps(r, _mm_mul_ps(lhs.col1_.get_simd(), y));
            r = _mm_add_ps(r, _mm_mul_ps(lhs.col2_.get_simd(), z));
            r = _mm_add_ps(r, _mm_mul_ps(lhs.col3_.get_simd(), w));

            res_cols[i].set_simd(r);
        }
        return res;
    }

    // ============================================================================
    // Useful Constants
    // ============================================================================

    inline const float4x4 float4x4_Identity = float4x4::identity();
    inline const float4x4 float4x4_Zero = float4x4::zero();
}

#endif // MATH_FLOAT4X4_INL
