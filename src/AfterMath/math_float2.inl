// Author: NSDeathman, DeepSeek
#pragma once
#include <emmintrin.h>

namespace AfterMath {

    // Исправленная функция загрузки
    inline __m128 _load_float2_fast(const float* ptr) {
        return _mm_set_ps(0.0f, 0.0f, ptr[1], ptr[0]);
    }

    // --- Constructors ---
    inline float2::float2(const float* data) noexcept : x(data[0]), y(data[1]) {}

    inline float2::float2(__m128 simd_) noexcept {
        alignas(16) float data[4];
        _mm_store_ps(data, simd_);
        x = data[0]; y = data[1];
    }

    // --- Assignment ---
    inline float2& float2::operator=(float scalar) noexcept {
        x = scalar;
        y = scalar;
        return *this;
    }

    // --- Compound Operators ---
    inline float2& float2::operator+=(const float2& rhs) noexcept {
        x += rhs.x;
        y += rhs.y;
        return *this;
    }

    inline float2& float2::operator-=(const float2& rhs) noexcept {
        x -= rhs.x;
        y -= rhs.y;
        return *this;
    }

    inline float2& float2::operator*=(const float2& rhs) noexcept {
        x *= rhs.x;
        y *= rhs.y;
        return *this;
    }

    inline float2& float2::operator/=(const float2& rhs) noexcept {
        x /= rhs.x;
        y /= rhs.y;
        return *this;
    }

    inline float2& float2::operator*=(float scalar) noexcept {
        x *= scalar;
        y *= scalar;
        return *this;
    }

    inline float2& float2::operator/=(float scalar) noexcept {
        float inv = 1.0f / scalar;
        x *= inv;
        y *= inv;
        return *this;
    }

    // --- Binary Operators ---
    inline float2 float2::operator+(const float2& rhs) const noexcept {
        return float2(x + rhs.x, y + rhs.y);
    }

    inline float2 float2::operator-(const float2& rhs) const noexcept {
        return float2(x - rhs.x, y - rhs.y);
    }

    inline float2 float2::operator+(const float& rhs) const noexcept {
        return float2(x + rhs, y + rhs);
    }

    inline float2 float2::operator-(const float& rhs) const noexcept {
        return float2(x - rhs, y - rhs);
    }

    // --- Access Operators ---
    inline float& float2::operator[](int index) noexcept {
        assert(index >= 0 && index < 2);
        return (&x)[index];
    }

    inline const float& float2::operator[](int index) const noexcept {
        assert(index >= 0 && index < 2);
        return (&x)[index];
    }

    inline float2::operator const float* () const noexcept { return &x; }
    inline float2::operator float* () noexcept { return &x; }
    inline float2::operator __m128() const noexcept {
        return _mm_set_ps(0.0f, 0.0f, y, x);
    }

    // --- Mathematical Functions ---
    inline float float2::length() const noexcept {
        return std::sqrt(x * x + y * y);
    }

    inline float2 float2::normalize() const noexcept {
        float len = length();
        if (len < Constants::Constants<float>::Epsilon) {
            return float2::zero();
        }
        float inv_len = 1.0f / len;
        return float2(x * inv_len, y * inv_len);
    }

    inline float float2::dot(const float2& other) const noexcept {
        return x * other.x + y * other.y;
    }

    inline float float2::cross(const float2& other) const {
        return x * other.y - y * other.x;
    }

    inline float float2::distance(const float2& other) const noexcept {
        return (*this - other).length();
    }

    // --- HLSL-like Functions ---
    inline float2 float2::abs() const noexcept {
        return float2(std::abs(x), std::abs(y));
    }

    inline float2 float2::sign() const noexcept {
        return float2(
            (x > 0.0f) ? 1.0f : ((x < 0.0f) ? -1.0f : 0.0f),
            (y > 0.0f) ? 1.0f : ((y < 0.0f) ? -1.0f : 0.0f)
        );
    }

    inline float2 float2::floor() const noexcept {
        return float2(std::floor(x), std::floor(y));
    }

    inline float2 float2::ceil() const noexcept {
        return float2(std::ceil(x), std::ceil(y));
    }

    inline float2 float2::round() const noexcept {
        return float2(std::round(x), std::round(y));
    }

    inline float2 float2::frac() const noexcept {
        return float2(x - std::floor(x), y - std::floor(y));
    }

    inline float2 float2::saturate() const noexcept {
        return float2(
            std::max(0.0f, std::min(1.0f, x)),
            std::max(0.0f, std::min(1.0f, y))
        );
    }

    inline float2 float2::step(float edge) const noexcept {
        return float2(
            (x >= edge) ? 1.0f : 0.0f,
            (y >= edge) ? 1.0f : 0.0f
        );
    }

    inline float2 float2::smoothstep(float edge0, float edge1) const noexcept {
        auto smooth = [edge0, edge1](float t) {
            t = std::max(0.0f, std::min(1.0f, (t - edge0) / (edge1 - edge0)));
            return t * t * (3.0f - 2.0f * t);
        };
        return float2(smooth(x), smooth(y));
    }

    // --- Geometric Operations ---
    inline float2 float2::reflect(const float2& normal) const noexcept {
        return *this - 2.0f * dot(normal) * normal;
    }

    inline float2 float2::refract(const float2& normal, float eta) const noexcept {
        // eta = n_incident / n_transmitted
        float cos_theta_i = -this->dot(normal);  // Косинус угла падения

        float sin_theta_i_sq = 1.0f - cos_theta_i * cos_theta_i;
        float sin_theta_t_sq = (eta * eta) * sin_theta_i_sq;

        // Полное внутреннее отражение
        if (sin_theta_t_sq > 1.0f) {
            return float2::zero();
        }

        float cos_theta_t = std::sqrt(1.0f - sin_theta_t_sq);
        return eta * (*this) + (eta * cos_theta_i - cos_theta_t) * normal;
    }

    inline float2 float2::rotate(float angle) const noexcept {
        float s = std::sin(angle);
        float c = std::cos(angle);
        return float2(x * c - y * s, x * s + y * c);
    }

    inline float float2::angle() const noexcept {
        return std::atan2(y, x);
    }

    // --- Utility Methods ---
    inline bool float2::isValid() const noexcept {
        return std::isfinite(x) && std::isfinite(y);
    }

    inline bool float2::approximately(const float2& other, float epsilon) const noexcept {
        return std::abs(x - other.x) <= epsilon && std::abs(y - other.y) <= epsilon;
    }

    inline bool float2::approximately_zero(float epsilon) const noexcept {
        return length_sq() <= epsilon * epsilon;
    }

    inline bool float2::is_normalized(float epsilon) const noexcept {
        return std::abs(length_sq() - 1.0f) <= epsilon;
    }

    inline std::string float2::to_string() const {
        char buf[64];
        std::snprintf(buf, 64, "(%.3f, %.3f)", x, y);
        return std::string(buf);
    }

    inline const float* float2::data() const noexcept { return &x; }
    inline float* float2::data() noexcept { return &x; }

    // --- Comparison Operators ---
    inline bool float2::operator==(const float2& rhs) const noexcept {
        return approximately(rhs);
    }

    inline bool float2::operator!=(const float2& rhs) const noexcept {
        return !(*this == rhs);
    }

    // --- Global Operators ---
    inline float2 operator*(float2 lhs, const float2& rhs) noexcept {
        return float2(lhs.x * rhs.x, lhs.y * rhs.y);
    }

    inline float2 operator/(float2 lhs, const float2& rhs) noexcept {
        return float2(lhs.x / rhs.x, lhs.y / rhs.y);
    }

    inline float2 operator*(float2 vec, float scalar) noexcept {
        return float2(vec.x * scalar, vec.y * scalar);
    }

    inline float2 operator*(float scalar, float2 vec) noexcept {
        return vec * scalar;
    }

    inline float2 operator/(float2 vec, float scalar) noexcept {
        return vec * (1.0f / scalar);
    }

    inline float2 operator+(float scalar, float2 vec) noexcept {
        return vec + scalar;
    }

    // --- Global Functions ---
    inline float distance(const float2& a, const float2& b) noexcept {
        return a.distance(b);
    }

    inline float distance_sq(const float2& a, const float2& b) noexcept {
        return a.distance_sq(b);
    }

    inline float dot(const float2& a, const float2& b) noexcept {
        return a.dot(b);
    }

    inline float cross(const float2& a, const float2& b) noexcept {
        return a.cross(b);
    }

    inline bool approximately(const float2& a, const float2& b, float e) noexcept {
        return a.approximately(b, e);
    }

    inline bool isValid(const float2& v) noexcept {
        return v.isValid();
    }

    inline float2 lerp(const float2& a, const float2& b, float t) noexcept {
        return a + (b - a) * t;
    }

    inline float2 slerp(const float2& a, const float2& b, float t) noexcept {
        // Нормализуем входные векторы
        float2 na = a.normalize();
        float2 nb = b.normalize();

        // Вычисляем угол между векторами
        float dot_val = dot(na, nb);
        dot_val = std::max(-1.0f, std::min(1.0f, dot_val));

        float theta = std::acos(dot_val) * t;
        float2 relative_vec = (nb - na * dot_val).normalize();

        return na * std::cos(theta) + relative_vec * std::sin(theta);
    }

    inline float2 perpendicular(const float2& v) noexcept {
        return v.perpendicular();
    }

    inline float2 reflect(const float2& incident, const float2& normal) noexcept {
        return incident.reflect(normal);
    }

    inline float2 refract(const float2& incident, const float2& normal, float eta) noexcept {
        return incident.refract(normal, eta);
    }

    inline float2 rotate(const float2& v, float angle) noexcept {
        return v.rotate(angle);
    }

    inline float angle_between(const float2& a, const float2& b) noexcept {
        float2 na = a.normalize();
        float2 nb = b.normalize();
        float dot_val = dot(na, nb);
        dot_val = std::max(-1.0f, std::min(1.0f, dot_val));
        return std::acos(dot_val);
    }

    inline float signed_angle_between(const float2& from, const float2& to) noexcept {
        float2 nfrom = from.normalize();
        float2 nto = to.normalize();
        float angle = std::atan2(cross(nfrom, nto), dot(nfrom, nto));
        return angle;
    }

    inline float2 project(const float2& v, const float2& onto) noexcept {
        float len_sq = onto.length_sq();
        if (len_sq < Constants::Constants<float>::Epsilon) {
            return float2::zero();
        }
        return onto * (dot(v, onto) / len_sq);
    }

    inline float2 reject(const float2& v, const float2& from) noexcept {
        return v - project(v, from);
    }

    // --- HLSL Global Wrappers ---
    inline float2 abs(const float2& v) noexcept { return v.abs(); }
    inline float2 sign(const float2& v) noexcept { return v.sign(); }
    inline float2 floor(const float2& v) noexcept { return v.floor(); }
    inline float2 ceil(const float2& v) noexcept { return v.ceil(); }
    inline float2 round(const float2& v) noexcept { return v.round(); }
    inline float2 frac(const float2& v) noexcept { return v.frac(); }
    inline float2 saturate(const float2& v) noexcept { return v.saturate(); }
    inline float2 step(float edge, const float2& v) noexcept { return v.step(edge); }
    inline float2 smoothstep(float edge0, float edge1, const float2& v) noexcept {
        return v.smoothstep(edge0, edge1);
    }

    inline float2 min(const float2& a, const float2& b) noexcept {
        return float2(std::min(a.x, b.x), std::min(a.y, b.y));
    }

    inline float2 max(const float2& a, const float2& b) noexcept {
        return float2(std::max(a.x, b.x), std::max(a.y, b.y));
    }

    inline float2 clamp(const float2& v, const float2& min_val, const float2& max_val) noexcept {
        return min(max(v, min_val), max_val);
    }

    // --- Utils ---
    inline float distance_to_line_segment(const float2& point,
        const float2& line_start,
        const float2& line_end) noexcept {
        float2 line_vec = line_end - line_start;
        float2 point_vec = point - line_start;

        float line_length_sq = line_vec.length_sq();
        if (line_length_sq < Constants::Constants<float>::Epsilon) {
            return distance(point, line_start);
        }

        float t = dot(point_vec, line_vec) / line_length_sq;
        t = std::max(0.0f, std::min(1.0f, t));

        float2 projection = line_start + line_vec * t;
        return distance(point, projection);
    }

    // --- Constants ---
    inline const float2 float2_Zero(0.0f, 0.0f);
    inline const float2 float2_One(1.0f, 1.0f);
    inline const float2 float2_UnitX(1.0f, 0.0f);
    inline const float2 float2_UnitY(0.0f, 1.0f);
    inline const float2 float2_Right(1.0f, 0.0f);
    inline const float2 float2_Left(-1.0f, 0.0f);
    inline const float2 float2_Up(0.0f, 1.0f);
    inline const float2 float2_Down(0.0f, -1.0f);
}
