#pragma once

#include <cmath>
#include <string>
#include <cstdio>
#include <xmmintrin.h>
#include <pmmintrin.h>
#include <smmintrin.h>

#include "math_config.h"
#include "math_constants.h"
#include "math_functions.h"
#include "math_fast_functions.h"
#include "math_float3.h"
#include "math_float4.h"
#include "math_float3x3.h"
#include "math_float4x4.h"

namespace Math
{
    class quaternion;

    quaternion operator+(quaternion lhs, const quaternion& rhs) noexcept;
    quaternion operator-(quaternion lhs, const quaternion& rhs) noexcept;
    quaternion operator*(const quaternion& lhs, const quaternion& rhs) noexcept;
    quaternion operator*(quaternion q, float scalar) noexcept;
    quaternion operator*(float scalar, quaternion q) noexcept;
    quaternion operator/(quaternion q, float scalar) noexcept;
    float3 operator*(const quaternion& q, const float3& vec) noexcept;

    quaternion nlerp(const quaternion& a, const quaternion& b, float t) noexcept;
    quaternion slerp(const quaternion& a, const quaternion& b, float t) noexcept;
    quaternion lerp(const quaternion& a, const quaternion& b, float t) noexcept;

    class quaternion
    {
    public:
        union {
            struct {
                float x;
                float y;
                float z;
                float w;
            };
            float4 data_;
            __m128 simd_;
        };

    public:
        quaternion() noexcept : data_(0.0f, 0.0f, 0.0f, 1.0f) {}
        quaternion(float x, float y, float z, float w) noexcept : data_(x, y, z, w) {}
        explicit quaternion(const float4& vec) noexcept : data_(vec) {}
        explicit quaternion(__m128 simd_val) noexcept : simd_(simd_val) {}

        quaternion(const float3& axis, float angle) noexcept
        {
            float3 normalized_axis = axis.normalize();
            float half_angle = angle * 0.5f;
            float sin_half = std::sin(half_angle);
            float cos_half = std::cos(half_angle);

            data_ = float4(normalized_axis * sin_half, cos_half);
        }

        quaternion(float pitch, float yaw, float roll) noexcept
        {
            float half_pitch = pitch * 0.5f;
            float half_yaw = yaw * 0.5f;
            float half_roll = roll * 0.5f;

            float cy = std::cos(half_yaw);
            float sy = std::sin(half_yaw);
            float cp = std::cos(half_pitch);
            float sp = std::sin(half_pitch);
            float cr = std::cos(half_roll);
            float sr = std::sin(half_roll);

            w = cr * cp * cy + sr * sp * sy;
            x = cr * sp * cy + sr * cp * sy;
            y = cr * cp * sy - sr * sp * cy;
            z = sr * cp * cy - cr * sp * sy;

            *this = normalize();
        }

        explicit quaternion(const float3x3& matrix) noexcept
        {
            *this = from_matrix(matrix);
        }

        explicit quaternion(const float4x4& matrix) noexcept
        {
            *this = from_matrix(matrix);
        }

        quaternion(const quaternion&) noexcept = default;

        quaternion& operator=(const quaternion&) noexcept = default;
        quaternion& operator=(const float4& vec) noexcept
        {
            data_ = vec;
            return *this;
        }

        static quaternion identity() noexcept
        {
            return quaternion(0.0f, 0.0f, 0.0f, 1.0f);
        }

        static quaternion zero() noexcept
        {
            return quaternion(0.0f, 0.0f, 0.0f, 0.0f);
        }

        static quaternion one() noexcept
        {
            return quaternion(1.0f, 1.0f, 1.0f, 1.0f);
        }

        bool approximately_zero(float epsilon = Constants::Constants<float>::Epsilon) const noexcept
        {
            return data_.approximately_zero(epsilon);
        }

        bool approximately(const quaternion& other, float epsilon = Constants::Constants<float>::Epsilon) const noexcept
        {
            float dot_val = dot(other);
            return (std::abs(dot_val) > (1.0f - epsilon));
        }

        static quaternion from_axis_angle(const float3& axis, float angle) noexcept
        {
            return quaternion(axis, angle);
        }

        static quaternion from_euler(float pitch, float yaw, float roll) noexcept
        {
            return quaternion(pitch, yaw, roll);
        }

        static quaternion from_euler(const float3& euler_angles) noexcept
        {
            return quaternion(euler_angles.x, euler_angles.y, euler_angles.z);
        }

        static quaternion from_matrix(const float3x3& matrix) noexcept
        {
            const float m00 = matrix(0, 0), m01 = matrix(0, 1), m02 = matrix(0, 2);
            const float m10 = matrix(1, 0), m11 = matrix(1, 1), m12 = matrix(1, 2);
            const float m20 = matrix(2, 0), m21 = matrix(2, 1), m22 = matrix(2, 2);

            const float epsilon = Constants::Constants<float>::Epsilon;
            const float epsilon_sq = epsilon * epsilon;

            float trace = m00 + m11 + m22;
            quaternion q;

            if (trace > epsilon) {
                float s = 0.5f / std::sqrt(trace + 1.0f);
                q.w = 0.25f / s;
                q.x = (m21 - m12) * s;
                q.y = (m02 - m20) * s;
                q.z = (m10 - m01) * s;
            }
            else {
                // След <= 0, находим наибольший диагональный элемент
                if (m00 > m11 && m00 > m22) {
                    float s = 2.0f * std::sqrt(1.0f + m00 - m11 - m22);
                    if (s > epsilon) {
                        float inv_s = 1.0f / s;
                        q.w = (m21 - m12) * inv_s;
                        q.x = 0.25f * s;
                        q.y = (m01 + m10) * inv_s;
                        q.z = (m02 + m20) * inv_s;
                    }
                    else {
                        // Матрица близка к единичной
                        return quaternion::identity();
                    }
                }
                else if (m11 > m22) {
                    float s = 2.0f * std::sqrt(1.0f + m11 - m00 - m22);
                    if (s > epsilon) {
                        float inv_s = 1.0f / s;
                        q.w = (m02 - m20) * inv_s;
                        q.x = (m01 + m10) * inv_s;
                        q.y = 0.25f * s;
                        q.z = (m12 + m21) * inv_s;
                    }
                    else {
                        return quaternion::identity();
                    }
                }
                else {
                    float s = 2.0f * std::sqrt(1.0f + m22 - m00 - m11);
                    if (s > epsilon) {
                        float inv_s = 1.0f / s;
                        q.w = (m10 - m01) * inv_s;
                        q.x = (m02 + m20) * inv_s;
                        q.y = (m12 + m21) * inv_s;
                        q.z = 0.25f * s;
                    }
                    else {
                        return quaternion::identity();
                    }
                }
            }

            // Нормализуем результат
            float len_sq = q.length_sq();
            if (len_sq > epsilon_sq && std::isfinite(len_sq)) {
                float inv_len = 1.0f / std::sqrt(len_sq);
                return quaternion(q.x * inv_len, q.y * inv_len, q.z * inv_len, q.w * inv_len);
            }

            return quaternion::identity();
        }

        static quaternion from_matrix(const float4x4& matrix) noexcept
        {
            float3x3 rot_matrix(
                matrix.row0().xyz(),
                matrix.row1().xyz(),
                matrix.row2().xyz()
            );
            return from_matrix(rot_matrix);
        }

        static quaternion from_to_rotation(const float3& from, const float3& to) noexcept
        {
            float3 v0 = from.normalize();
            float3 v1 = to.normalize();

            if (v0.approximately_zero() || v1.approximately_zero()) {
                return identity();
            }

            float cos_angle = float3::dot(v0, v1);

            if (cos_angle > 0.9999f) {
                return identity();
            }

            if (cos_angle < -0.9999f) {
                float3 axis = cross(float3::unit_x(), v0);
                if (axis.length_sq() < 0.0001f) {
                    axis = cross(float3::unit_y(), v0);
                }
                axis = axis.normalize();
                return quaternion(axis, Constants::PI);
            }

            float3 axis = cross(v0, v1);
            float s = std::sqrt((1.0f + cos_angle) * 2.0f);
            float inv_s = 1.0f / s;

            return quaternion(axis.x * inv_s, axis.y * inv_s, axis.z * inv_s, s * 0.5f).normalize();
        }

        static quaternion look_rotation(const float3& forward, const float3& up) noexcept
        {
            float3 f = forward.normalize();

            if (f.length_sq() < 1e-6f) {
                return identity();
            }

            float3 r = Math::cross(up, f).normalize();

            if (r.length_sq() < 1e-6f) {
                r = Math::cross(float3::unit_x(), f).normalize();
                if (r.length_sq() < 1e-6f) {
                    r = Math::cross(float3::unit_z(), f).normalize();
                }
            }

            float3 u = Math::cross(f, r);
            float3x3 rot_mat(r, u, f);

            return from_matrix(rot_mat);
        }

        static quaternion rotation_x(float angle) noexcept {
            return quaternion(float3::unit_x(), angle);
        }

        static quaternion rotation_y(float angle) noexcept {
            return quaternion(float3::unit_y(), angle);
        }

        static quaternion rotation_z(float angle) noexcept {
            float half_angle = angle * 0.5f;
            return quaternion(0.0f, 0.0f, std::sin(half_angle), std::cos(half_angle));
        }

        static quaternion slerp(const quaternion& a, const quaternion& b, float t) noexcept
        {
            return Math::slerp(a, b, t);
        }

        static quaternion lerp(const quaternion& a, const quaternion& b, float t) noexcept
        {
            return Math::lerp(a, b, t);
        }

        quaternion& operator+=(const quaternion& rhs) noexcept
        {
            data_ += rhs.data_;
            return *this;
        }

        quaternion& operator-=(const quaternion& rhs) noexcept
        {
            data_ -= rhs.data_;
            return *this;
        }

        quaternion& operator*=(float scalar) noexcept
        {
            data_ *= scalar;
            return *this;
        }

        quaternion& operator/=(float scalar) noexcept
        {
            data_ /= scalar;
            return *this;
        }

        quaternion& operator*=(const quaternion& rhs) noexcept
        {
            *this = *this * rhs;
            return *this;
        }

        quaternion operator+() const noexcept { return *this; }
        quaternion operator-() const noexcept { return quaternion(-data_); }
        operator float4() const noexcept { return data_; }

        __m128 get_simd() const noexcept { return simd_; }
        void set_simd(__m128 new_simd) noexcept { simd_ = new_simd; }

        float length() const noexcept { return data_.length(); }
        float length_sq() const noexcept { return data_.length_sq(); }

        quaternion normalize() const noexcept
        {
            float len_sq = length_sq();
            if (len_sq > Constants::Constants<float>::Epsilon && std::isfinite(len_sq)) {
                float inv_len = 1.0f / std::sqrt(len_sq);
                return quaternion(data_ * inv_len);
            }
            return identity();
        }

        quaternion conjugate() const noexcept
        {
            static const __m128 SIGN_MASK = _mm_set_ps(1.0f, -1.0f, -1.0f, -1.0f);
            return quaternion(_mm_mul_ps(simd_, SIGN_MASK));
        }

        quaternion inverse() const noexcept
        {
            float len_sq = length_sq();
            if (len_sq > Constants::Constants<float>::Epsilon) {
                return conjugate() / len_sq;
            }
            return identity();
        }

        float dot(const quaternion& other) const noexcept
        {
            return float4::dot(data_, other.data_);
        }

        float3x3 to_matrix3x3() const noexcept
        {
            quaternion n = normalize();

            float xx = n.x * n.x;
            float yy = n.y * n.y;
            float zz = n.z * n.z;
            float xy = n.x * n.y;
            float xz = n.x * n.z;
            float yz = n.y * n.z;
            float wx = n.w * n.x;
            float wy = n.w * n.y;
            float wz = n.w * n.z;

            return float3x3(
                float3(1.0f - 2.0f * (yy + zz), 2.0f * (xy - wz), 2.0f * (xz + wy)),
                float3(2.0f * (xy + wz), 1.0f - 2.0f * (xx + zz), 2.0f * (yz - wx)),
                float3(2.0f * (xz - wy), 2.0f * (yz + wx), 1.0f - 2.0f * (xx + yy))
            );
        }

        float4x4 to_matrix4x4() const noexcept
        {
            float3x3 rot = to_matrix3x3();
            return float4x4(
                float4(rot.row0(), 0.0f),
                float4(rot.row1(), 0.0f),
                float4(rot.row2(), 0.0f),
                float4(0.0f, 0.0f, 0.0f, 1.0f)
            );
        }

        void to_axis_angle(float3& axis, float& angle) const noexcept
        {
            quaternion normalized = normalize();

            angle = 2.0f * std::acos(std::clamp(normalized.w, -1.0f, 1.0f));

            if (angle < Constants::Constants<float>::Epsilon) {
                axis = float3::unit_x();
                return;
            }

            float sin_half_angle = std::sqrt(1.0f - normalized.w * normalized.w);

            if (sin_half_angle > Constants::Constants<float>::Epsilon) {
                float inv_sin = 1.0f / sin_half_angle;
                axis = float3(normalized.x * inv_sin, normalized.y * inv_sin, normalized.z * inv_sin);
                axis = axis.normalize();
            }
            else {
                axis = float3::unit_x();
            }
        }

        float3 to_euler() const noexcept
        {
            quaternion n = normalize();
            float x = n.x, y = n.y, z = n.z, w = n.w;

            float3 euler;

            float sinr_cosp = 2.0f * (w * x + y * z);
            float cosr_cosp = 1.0f - 2.0f * (x * x + y * y);
            euler.x = std::atan2(sinr_cosp, cosr_cosp);

            float sinp = 2.0f * (w * y - z * x);
            if (std::abs(sinp) >= 1.0f) {
                euler.y = std::copysign(Constants::HALF_PI, sinp);
            }
            else {
                euler.y = std::asin(sinp);
            }

            float siny_cosp = 2.0f * (w * z + x * y);
            float cosy_cosp = 1.0f - 2.0f * (y * y + z * z);
            euler.z = std::atan2(siny_cosp, cosy_cosp);

            return euler;
        }

        float3 transform_vector(const float3& vec) const noexcept
        {
            quaternion q = normalize();

            float3 q_xyz(q.x, q.y, q.z);
            float3 t = cross(q_xyz, vec) * 2.0f;
            float3 result = vec + q.w * t + cross(q_xyz, t);

            return result;
        }

        float3 transform_direction(const float3& dir) const noexcept
        {
            return transform_vector(dir).normalize();
        }

        bool is_identity(float epsilon = Constants::Constants<float>::Epsilon) const noexcept
        {
            return (std::abs(x) < epsilon) &&
                (std::abs(y) < epsilon) &&
                (std::abs(z) < epsilon) &&
                (std::abs(w - 1.0f) < epsilon);
        }

        bool is_normalized(float epsilon = Constants::Constants<float>::Epsilon) const noexcept
        {
            float len_sq = length_sq();

            if (!std::isfinite(len_sq)) {
                return false;
            }

            if (len_sq < Constants::Constants<float>::Epsilon) {
                return false;
            }

            return MathFunctions::approximately(len_sq, 1.0f, epsilon);
        }

        bool is_valid() const noexcept
        {
            return data_.isValid();
        }

        std::string to_string() const
        {
            char buffer[256];
            std::snprintf(buffer, sizeof(buffer), "(%.3f, %.3f, %.3f, %.3f)", x, y, z, w);
            return std::string(buffer);
        }

        const float* data() const noexcept { return &x; }
        float* data() noexcept { return &x; }
        const float4& get_float4() const noexcept { return data_; }
        float4& get_float4() noexcept { return data_; }

        bool operator==(const quaternion& rhs) const noexcept { return approximately(rhs); }
        bool operator!=(const quaternion& rhs) const noexcept { return !approximately(rhs); }

        quaternion fast_normalize() const noexcept
        {
            float len_sq = length_sq();

            if (len_sq > 1e-12f && std::isfinite(len_sq)) {
                float inv_len = FastMath::fast_inv_sqrt(len_sq);
                inv_len = inv_len * (1.5f - 0.5f * len_sq * inv_len * inv_len);
                return quaternion(data_ * inv_len);
            }
            return identity();
        }
    };

    inline quaternion operator+(quaternion lhs, const quaternion& rhs) noexcept
    {
        return lhs += rhs;
    }

    inline quaternion operator-(quaternion lhs, const quaternion& rhs) noexcept
    {
        return lhs -= rhs;
    }

    inline quaternion operator*(const quaternion& lhs, const quaternion& rhs) noexcept
    {
        return quaternion(
            lhs.w * rhs.x + lhs.x * rhs.w + lhs.y * rhs.z - lhs.z * rhs.y,
            lhs.w * rhs.y - lhs.x * rhs.z + lhs.y * rhs.w + lhs.z * rhs.x,
            lhs.w * rhs.z + lhs.x * rhs.y - lhs.y * rhs.x + lhs.z * rhs.w,
            lhs.w * rhs.w - lhs.x * rhs.x - lhs.y * rhs.y - lhs.z * rhs.z
        );
    }

    inline quaternion operator*(quaternion q, float scalar) noexcept
    {
        return q *= scalar;
    }

    inline quaternion operator*(float scalar, quaternion q) noexcept
    {
        return q *= scalar;
    }

    inline quaternion operator/(quaternion q, float scalar) noexcept
    {
        return q /= scalar;
    }

    inline float3 operator*(const quaternion& q, const float3& vec) noexcept
    {
        return q.transform_vector(vec);
    }

    inline float length(const quaternion& q) noexcept { return q.length(); }
    inline float length_sq(const quaternion& q) noexcept { return q.length_sq(); }
    inline quaternion normalize(const quaternion& q) noexcept { return q.normalize(); }
    inline quaternion conjugate(const quaternion& q) noexcept { return q.conjugate(); }
    inline quaternion inverse(const quaternion& q) noexcept { return q.inverse(); }
    inline float dot(const quaternion& a, const quaternion& b) noexcept { return a.dot(b); }

    inline quaternion nlerp(const quaternion& a, const quaternion& b, float t) noexcept
    {
        float cos_angle = dot(a, b);
        float sign = (cos_angle < 0.0f) ? -1.0f : 1.0f;

        __m128 a_simd = a.get_simd();
        __m128 b_simd = _mm_mul_ps(b.get_simd(), _mm_set1_ps(sign));
        __m128 t_vec = _mm_set1_ps(t);
        __m128 one_minus_t = _mm_set1_ps(1.0f - t);

        __m128 result = _mm_add_ps(_mm_mul_ps(a_simd, one_minus_t), _mm_mul_ps(b_simd, t_vec));

        __m128 len_sq = _mm_dp_ps(result, result, 0xFF);
        __m128 rsqrt = _mm_rsqrt_ps(len_sq);

        __m128 half_x = _mm_mul_ps(len_sq, _mm_set1_ps(0.5f));
        __m128 rsqrt_sq = _mm_mul_ps(rsqrt, rsqrt);
        __m128 adj = _mm_sub_ps(_mm_set1_ps(1.5f), _mm_mul_ps(half_x, rsqrt_sq));
        __m128 precise_inv_len = _mm_mul_ps(rsqrt, adj);

        result = _mm_mul_ps(result, precise_inv_len);

        return quaternion(result);
    }

    inline quaternion slerp(const quaternion& a, const quaternion& b, float t) noexcept
    {
        if (t <= 0.0f) return a.normalize();
        if (t >= 1.0f) return b.normalize();

        float cos_angle = dot(a, b);
        quaternion b_target = b;

        // Если угол больше 90°, используем отрицательный кватернион для кратчайшего пути
        if (cos_angle < 0.0f) {
            b_target = -b;
            cos_angle = -cos_angle;
        }

        const float THRESHOLD = 0.9995f;

        // Если кватернионы почти параллельны, используем линейную интерполяцию
        if (cos_angle > THRESHOLD) {
            return nlerp(a, b_target, t);
        }

        // Ограничиваем cos_angle для численной устойчивости
        cos_angle = std::min(std::max(cos_angle, -1.0f), 1.0f);

        // Вычисляем угол и коэффициенты для сферической интерполяции
        float angle = std::acos(cos_angle);
        float sin_angle = std::sin(angle);

        // Проверка на очень маленький sin_angle (почти параллельные векторы)
        if (std::abs(sin_angle) < Constants::Constants<float>::Epsilon) {
            return nlerp(a, b_target, t);
        }

        float inv_sin = 1.0f / sin_angle;
        float ratio_a = std::sin((1.0f - t) * angle) * inv_sin;
        float ratio_b = std::sin(t * angle) * inv_sin;

        quaternion result = (a * ratio_a + b_target * ratio_b);

        // Гарантируем, что результат нормализован (всегда для slerp)
        return result.normalize();
    }

    inline quaternion lerp(const quaternion& a, const quaternion& b, float t) noexcept
    {
        return nlerp(a, b, t);
    }

    inline bool approximately(const quaternion& a, const quaternion& b, float epsilon = Constants::Constants<float>::Epsilon) noexcept
    {
        return a.approximately(b, epsilon);
    }

    inline bool is_valid(const quaternion& q) noexcept { return q.is_valid(); }
    inline bool is_normalized(const quaternion& q, float epsilon = Constants::Constants<float>::Epsilon) noexcept { return q.is_normalized(epsilon); }

    inline const quaternion quaternion_Identity = quaternion::identity();
    inline const quaternion quaternion_Zero = quaternion::zero();
    inline const quaternion quaternion_One = quaternion::one();
}
