#ifndef MATH_FLOAT2X2_INL
#define MATH_FLOAT2X2_INL

namespace Math
{
    // ============================================================================
    // Constructors
    // ============================================================================

    inline float2x2::float2x2() noexcept
        : m00(1.0f), m10(0.0f), m01(0.0f), m11(1.0f) {}

    inline float2x2::float2x2(const float2& col0, const float2& col1) noexcept
        : m00(col0.x), m10(col0.y), m01(col1.x), m11(col1.y) {}

    inline float2x2::float2x2(float m00, float m10, float m01, float m11) noexcept
        : m00(m00), m10(m10), m01(m01), m11(m11) {}

    inline float2x2::float2x2(const float* data) noexcept
        : m00(data[0]), m10(data[1]), m01(data[2]), m11(data[3]) {}

    inline float2x2::float2x2(float scalar) noexcept
        : m00(scalar), m10(0.0f), m01(0.0f), m11(scalar) {}

    inline float2x2::float2x2(const float2& diagonal) noexcept
        : m00(diagonal.x), m10(0.0f), m01(0.0f), m11(diagonal.y) {}

    inline float2x2::float2x2(__m128 sse_data) noexcept
    {
        set_sse_data(sse_data);
    }

    // ============================================================================
    // Static Constructors
    // ============================================================================

    inline float2x2 float2x2::identity() noexcept
    {
        return float2x2(1.0f, 0.0f, 0.0f, 1.0f);
    }

    inline float2x2 float2x2::zero() noexcept
    {
        return float2x2(0.0f, 0.0f, 0.0f, 0.0f);
    }

    inline float2x2 float2x2::rotation(float angle) noexcept
    {
        float s, c;
        MathFunctions::sin_cos(angle, &s, &c);
        return float2x2(c, s, -s, c);
    }

    inline float2x2 float2x2::scaling(const float2& scale) noexcept
    {
        return float2x2(scale.x, 0.0f, 0.0f, scale.y);
    }

    inline float2x2 float2x2::scaling(float x, float y) noexcept
    {
        return float2x2(x, 0.0f, 0.0f, y);
    }

    inline float2x2 float2x2::scaling(float uniformScale) noexcept
    {
        return float2x2(uniformScale, 0.0f, 0.0f, uniformScale);
    }

    inline float2x2 float2x2::shear(const float2& shear) noexcept
    {
        return float2x2(1.0f, shear.y, shear.x, 1.0f);
    }

    inline float2x2 float2x2::shear(float x, float y) noexcept
    {
        return float2x2(1.0f, y, x, 1.0f);
    }

    inline float2x2 float2x2::reflection(const float2& axis) noexcept
    {
        // R = I - 2 * axis * axis^T
        float2 n = axis.normalize();
        float xx = n.x * n.x;
        float yy = n.y * n.y;
        float xy = n.x * n.y;

        return float2x2(
            1.0f - 2.0f * xx,
            -2.0f * xy,
            -2.0f * xy,
            1.0f - 2.0f * yy
        );
    }

    inline float2x2 float2x2::orthonormal_basis_from_x(const float2& x_axis) noexcept
    {
        float2 x = x_axis.normalize();
        float2 y = float2(-x.y, x.x);  // Perpendicular vector (rotated 90 degrees CCW)
        return float2x2(x, y);
    }

    inline float2x2 float2x2::orthonormal_basis_from_y(const float2& y_axis) noexcept
    {
        float2 y = y_axis.normalize();
        float2 x = float2(y.y, -y.x);  // Perpendicular vector (rotated 90 degrees CW)
        return float2x2(x, y);
    }

    // ============================================================================
    // Access Operators (FIXED - Safe version)
    // ============================================================================

    inline float& float2x2::operator()(int row, int col) noexcept
    {
        MATH_ASSERT(row >= 0 && row < 2);
        MATH_ASSERT(col >= 0 && col < 2);
        return data[col * 2 + row];
    }

    inline const float& float2x2::operator()(int row, int col) const noexcept
    {
        MATH_ASSERT(row >= 0 && row < 2);
        MATH_ASSERT(col >= 0 && col < 2);
        return data[col * 2 + row];
    }

    inline float2 float2x2::col(int colIndex) const noexcept
    {
        MATH_ASSERT(colIndex >= 0 && colIndex < 2);
        return colIndex == 0 ? col0() : col1();
    }

    inline float2 float2x2::row(int rowIndex) const noexcept
    {
        MATH_ASSERT(rowIndex >= 0 && rowIndex < 2);
        return rowIndex == 0 ? row0() : row1();
    }

    // ============================================================================
    // Column and Row Accessors
    // ============================================================================

    inline float2 float2x2::col0() const noexcept { return float2(m00, m10); }
    inline float2 float2x2::col1() const noexcept { return float2(m01, m11); }

    inline float2 float2x2::row0() const noexcept { return float2(m00, m01); }
    inline float2 float2x2::row1() const noexcept { return float2(m10, m11); }

    inline void float2x2::set_col0(const float2& col) noexcept
    {
        m00 = col.x;
        m10 = col.y;
    }

    inline void float2x2::set_col1(const float2& col) noexcept
    {
        m01 = col.x;
        m11 = col.y;
    }

    inline void float2x2::set_row0(const float2& row) noexcept
    {
        m00 = row.x;
        m01 = row.y;
    }

    inline void float2x2::set_row1(const float2& row) noexcept
    {
        m10 = row.x;
        m11 = row.y;
    }

    // ============================================================================
    // SSE Accessors
    // ============================================================================

    inline __m128 float2x2::sse_data() const noexcept
    {
        return _mm_load_ps(data);
    }

    inline void float2x2::set_sse_data(__m128 sse_data) noexcept
    {
        _mm_store_ps(data, sse_data);
    }

    // ============================================================================
    // Compound Assignment Operators
    // ============================================================================

    inline float2x2& float2x2::operator+=(const float2x2& rhs) noexcept
    {
        __m128 result = _mm_add_ps(sse_data(), rhs.sse_data());
        set_sse_data(result);
        return *this;
    }

    inline float2x2& float2x2::operator-=(const float2x2& rhs) noexcept
    {
        __m128 result = _mm_sub_ps(sse_data(), rhs.sse_data());
        set_sse_data(result);
        return *this;
    }

    inline float2x2& float2x2::operator*=(float scalar) noexcept
    {
        __m128 scale = _mm_set1_ps(scalar);
        __m128 result = _mm_mul_ps(sse_data(), scale);
        set_sse_data(result);
        return *this;
    }

    inline float2x2& float2x2::operator/=(float scalar) noexcept
    {
        MATH_ASSERT(scalar != 0.0f);
        __m128 inv_scale = _mm_set1_ps(1.0f / scalar);
        __m128 result = _mm_mul_ps(sse_data(), inv_scale);
        set_sse_data(result);
        return *this;
    }

    inline float2x2& float2x2::operator*=(const float2x2& rhs) noexcept
    {
        *this = *this * rhs;
        return *this;
    }

    // ============================================================================
    // Unary Operators
    // ============================================================================

    inline float2x2 float2x2::operator+() const noexcept { return *this; }

    inline float2x2 float2x2::operator-() const noexcept
    {
        __m128 neg = _mm_sub_ps(_mm_setzero_ps(), sse_data());
        return float2x2(neg);
    }

    // ============================================================================
    // Matrix Operations
    // ============================================================================

    inline float2x2 float2x2::transposed() const noexcept
    {
        __m128 v = sse_data();
        __m128 swapped = _mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 1, 2, 0));
        return float2x2(swapped);
    }

    inline float float2x2::determinant() const noexcept
    {
        return m00 * m11 - m10 * m01;
    }

    inline float2x2 float2x2::inverted() const noexcept
    {
        const float det = determinant();
        if (std::abs(det) < Constants::Constants<float>::Epsilon)
        {
            return identity();
        }

        const float inv_det = 1.0f / det;
        return float2x2(
            m11 * inv_det,
            -m10 * inv_det,
            -m01 * inv_det,
            m00 * inv_det
        );
    }

    inline float2x2 float2x2::adjugate() const noexcept
    {
        return float2x2(m11, -m10, -m01, m00);
    }

    inline float float2x2::trace() const noexcept
    {
        return m00 + m11;
    }

    inline float2 float2x2::diagonal() const noexcept
    {
        return float2(m00, m11);
    }

    inline float float2x2::frobenius_norm() const noexcept
    {
        return std::sqrt(m00 * m00 + m10 * m10 + m01 * m01 + m11 * m11);
    }

    // ============================================================================
    // SSE-optimized Matrix Multiplication
    // ============================================================================

    inline float2x2 operator*(const float2x2& lhs, const float2x2& rhs) noexcept
    {
        // SSE-optimized version
        __m128 lhs_data = lhs.sse_data();
        __m128 rhs_data = rhs.sse_data();

        // Extract and broadcast columns of rhs
        __m128 rhs_col0 = _mm_shuffle_ps(rhs_data, rhs_data, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 rhs_col1 = _mm_shuffle_ps(rhs_data, rhs_data, _MM_SHUFFLE(1, 1, 1, 1));

        // Multiply lhs by broadcasted columns
        __m128 lhs_swizzle0 = _mm_shuffle_ps(lhs_data, lhs_data, _MM_SHUFFLE(1, 0, 1, 0));
        __m128 lhs_swizzle1 = _mm_shuffle_ps(lhs_data, lhs_data, _MM_SHUFFLE(3, 2, 3, 2));

        __m128 result_col0 = _mm_add_ps(
            _mm_mul_ps(lhs_swizzle0, rhs_col0),
            _mm_mul_ps(lhs_swizzle1, _mm_shuffle_ps(rhs_data, rhs_data, _MM_SHUFFLE(1, 1, 1, 1)))
        );

        __m128 result_col1 = _mm_add_ps(
            _mm_mul_ps(lhs_swizzle0, rhs_col1),
            _mm_mul_ps(lhs_swizzle1, _mm_shuffle_ps(rhs_data, rhs_data, _MM_SHUFFLE(3, 3, 3, 3)))
        );

        // Combine results
        __m128 result = _mm_shuffle_ps(result_col0, result_col1, _MM_SHUFFLE(1, 0, 1, 0));
        return float2x2(result);
    }

    // ============================================================================
    // Vector Transformations
    // ============================================================================

    inline float2 float2x2::transform_vector(const float2& vec) const noexcept
    {
        return float2(
            m00 * vec.x + m01 * vec.y,
            m10 * vec.x + m11 * vec.y
        );
    }

    inline float2 float2x2::transform_point(const float2& point) const noexcept
    {
        return transform_vector(point);
    }

    inline float2 float2x2::transform_vector_left(const float2& vec) const noexcept
    {
        return float2(
            vec.x * m00 + vec.y * m10,
            vec.x * m01 + vec.y * m11
        );
    }

    // ============================================================================
    // Decomposition Methods
    // ============================================================================

    inline bool float2x2::decompose_rotation_scale(float& rotation, float2& scale) const noexcept
    {
        // Extract scale from column lengths
        float2 col0 = this->col0();
        float2 col1 = this->col1();

        float len0 = col0.length();
        float len1 = col1.length();

        if (len0 < Constants::Constants<float>::Epsilon ||
            len1 < Constants::Constants<float>::Epsilon)
        {
            return false;
        }

        // Normalize first column to get rotation
        float2 normalized_col0 = col0 / len0;
        rotation = std::atan2(normalized_col0.y, normalized_col0.x);
        scale = float2(len0, len1);

        return true;
    }

    inline bool float2x2::decompose_rotation_scale_shear(float& rotation, float2& scale, float2& shear) const noexcept
    {
        // Perform QR decomposition (Gram-Schmidt)
        float2 col0 = this->col0();
        float2 col1 = this->col1();

        // First column is the first basis vector
        float len0 = col0.length();
        if (len0 < Constants::Constants<float>::Epsilon)
            return false;

        float2 q0 = col0 / len0;

        // Project second column onto first
        float proj = dot(col1, q0);
        float2 proj_vec = q0 * proj;

        // Orthogonal component
        float2 ortho = col1 - proj_vec;
        float len1 = ortho.length();

        if (len1 < Constants::Constants<float>::Epsilon)
            return false;

        float2 q1 = ortho / len1;

        // Extract components
        rotation = std::atan2(q0.y, q0.x);
        scale = float2(len0, len1);
        shear = float2(proj / len0, 0.0f);  // Only X shear in 2D

        return true;
    }

    // ============================================================================
    // Transformation Component Extraction (IMPROVED)
    // ============================================================================

    inline float float2x2::get_rotation() const noexcept
    {
        // Extract rotation from normalized first column
        float len = std::sqrt(m00 * m00 + m10 * m10);
        if (len < Constants::Constants<float>::Epsilon)
            return 0.0f;
        return std::atan2(m10 / len, m00 / len);
    }

    inline float2 float2x2::get_scale() const noexcept
    {
        return float2(
            std::sqrt(m00 * m00 + m10 * m10),
            std::sqrt(m01 * m01 + m11 * m11)
        );
    }

    inline float2 float2x2::get_shear() const noexcept
    {
        float2 col0 = this->col0();
        float2 col1 = this->col1();
        float len0_sq = col0.length_sq();
        if (len0_sq < Constants::Constants<float>::Epsilon)
            return float2(0.0f, 0.0f);

        float shear_x = dot(col1, col0) / len0_sq;
        return float2(shear_x, 0.0f);
    }

    inline void float2x2::set_rotation(float angle) noexcept
    {
        // Preserve only scale, discard shear
        float2 current_scale = get_scale();
        float cos_angle = std::cos(angle);
        float sin_angle = std::sin(angle);

        m00 = cos_angle * current_scale.x;
        m10 = sin_angle * current_scale.x;
        m01 = -sin_angle * current_scale.y;
        m11 = cos_angle * current_scale.y;
    }

    inline void float2x2::set_scale(const float2& scale) noexcept
    {
        // Reset to scaling matrix, discarding rotation and shear
        m00 = scale.x;
        m10 = 0.0f;
        m01 = 0.0f;
        m11 = scale.y;
    }

    inline void float2x2::set_scale_preserve_shear(const float2& scale) noexcept
    {
        // Preserve shear while changing scale
        float2 current_scale = get_scale();
        if (current_scale.x > 0)
        {
            float factor = scale.x / current_scale.x;
            m00 *= factor;
            m10 *= factor;
        }
        if (current_scale.y > 0)
        {
            float factor = scale.y / current_scale.y;
            m01 *= factor;
            m11 *= factor;
        }
    }

    inline void float2x2::set_shear(const float2& shear) noexcept
    {
        // Add shear to existing rotation/scale
        float2 current_shear = get_shear();
        float shear_diff = shear.x - current_shear.x;

        m01 += m00 * shear_diff;
        m11 += m10 * shear_diff;
    }

    // ============================================================================
    // Utility Methods
    // ============================================================================

    inline bool float2x2::is_identity(float epsilon) const noexcept
    {
        return MathFunctions::approximately(m00, 1.0f, epsilon) &&
            MathFunctions::approximately(m10, 0.0f, epsilon) &&
            MathFunctions::approximately(m01, 0.0f, epsilon) &&
            MathFunctions::approximately(m11, 1.0f, epsilon);
    }

    inline bool float2x2::is_orthogonal(float epsilon) const noexcept
    {
        float2 col0 = this->col0();
        float2 col1 = this->col1();
        return MathFunctions::approximately(dot(col0, col1), 0.0f, epsilon);
    }

    inline bool float2x2::is_rotation(float epsilon) const noexcept
    {
        float2 col0 = this->col0();
        float2 col1 = this->col1();
        return MathFunctions::approximately(col0.length_sq(), 1.0f, epsilon) &&
            MathFunctions::approximately(col1.length_sq(), 1.0f, epsilon) &&
            MathFunctions::approximately(dot(col0, col1), 0.0f, epsilon) &&
            MathFunctions::approximately(determinant(), 1.0f, epsilon);
    }

    inline bool float2x2::is_scale_uniform(float epsilon) const noexcept
    {
        float2 scale = get_scale();
        return MathFunctions::approximately(scale.x, scale.y, epsilon);
    }

    inline bool float2x2::approximately(const float2x2& other, float epsilon) const noexcept
    {
        return MathFunctions::approximately(m00, other.m00, epsilon) &&
            MathFunctions::approximately(m10, other.m10, epsilon) &&
            MathFunctions::approximately(m01, other.m01, epsilon) &&
            MathFunctions::approximately(m11, other.m11, epsilon);
    }

    inline bool float2x2::approximately_zero(float epsilon) const noexcept
    {
        return MathFunctions::approximately(m00, 0.0f, epsilon) &&
            MathFunctions::approximately(m10, 0.0f, epsilon) &&
            MathFunctions::approximately(m01, 0.0f, epsilon) &&
            MathFunctions::approximately(m11, 0.0f, epsilon);
    }

    inline std::string float2x2::to_string() const
    {
        char buffer[256];
        std::snprintf(buffer, sizeof(buffer),
            "[%8.4f, %8.4f]\n"
            "[%8.4f, %8.4f]",
            m00, m01,
            m10, m11);
        return std::string(buffer);
    }

    inline const float* float2x2::column_major_data() const noexcept
    {
        return data;
    }

    inline void float2x2::to_column_major(float* data) const noexcept
    {
        data[0] = m00;
        data[1] = m10;
        data[2] = m01;
        data[3] = m11;
    }

    inline void float2x2::to_row_major(float* data) const noexcept
    {
        data[0] = m00;
        data[1] = m01;
        data[2] = m10;
        data[3] = m11;
    }

    // ============================================================================
    // Comparison Operators
    // ============================================================================

    inline bool float2x2::operator==(const float2x2& rhs) const noexcept
    {
        return approximately(rhs);
    }

    inline bool float2x2::operator!=(const float2x2& rhs) const noexcept
    {
        return !(*this == rhs);
    }

    inline bool float2x2::operator<(const float2x2& rhs) const noexcept
    {
        // Lexicographic comparison for use in containers
        if (m00 != rhs.m00) return m00 < rhs.m00;
        if (m10 != rhs.m10) return m10 < rhs.m10;
        if (m01 != rhs.m01) return m01 < rhs.m01;
        return m11 < rhs.m11;
    }

    // ============================================================================
    // Specialized Operations
    // ============================================================================

    inline float2x2 float2x2::orthonormalized(float epsilon) const noexcept
    {
        // Gram-Schmidt process
        float2 col0 = this->col0();
        float2 col1 = this->col1();

        float len0 = col0.length();
        if (len0 < epsilon) return identity();

        float2 q0 = col0 / len0;

        float proj = dot(col1, q0);
        float2 q1 = col1 - q0 * proj;

        float len1 = q1.length();
        if (len1 < epsilon) return identity();

        q1 = q1 / len1;

        return float2x2(q0, q1);
    }

    inline float2x2 float2x2::closest_rotation() const noexcept
    {
        // Polar decomposition: R = M * (M^T * M)^{-1/2}
        float2x2 mt = this->transposed();
        float2x2 mtm = mt * (*this);

        // For 2x2, we can compute (M^T * M)^{-1/2} analytically
        float det = mtm.determinant();
        if (det < Constants::Constants<float>::Epsilon)
            return identity();

        float trace = mtm.trace();
        float s = std::sqrt(det);
        float t = std::sqrt(trace + 2.0f * s);

        float2x2 inverse_sqrt = (mtm + float2x2(s)) * (1.0f / (t * s));

        return (*this) * inverse_sqrt;
    }

    inline float2x2 float2x2::lerp(const float2x2& b, float t) const noexcept
    {
        return float2x2(
            MathFunctions::lerp(m00, b.m00, t),
            MathFunctions::lerp(m10, b.m10, t),
            MathFunctions::lerp(m01, b.m01, t),
            MathFunctions::lerp(m11, b.m11, t)
        );
    }

    inline float2x2 float2x2::slerp(const float2x2& b, float t) const noexcept
    {
        // Extract rotations and interpolate angles
        float angle_a = this->get_rotation();
        float angle_b = b.get_rotation();

        // Find shortest path
        float diff = angle_b - angle_a;
        if (diff > Constants::PI) diff -= 2.0f * Constants::PI;
        else if (diff < -Constants::PI) diff += 2.0f * Constants::PI;

        float interpolated_angle = angle_a + t * diff;
        return float2x2::rotation(interpolated_angle);
    }

    // ============================================================================
    // Binary Operators
    // ============================================================================

    inline float2x2 operator+(float2x2 lhs, const float2x2& rhs) noexcept
    {
        return lhs += rhs;
    }

    inline float2x2 operator-(float2x2 lhs, const float2x2& rhs) noexcept
    {
        return lhs -= rhs;
    }

    inline float2x2 operator*(float2x2 mat, float scalar) noexcept
    {
        return mat *= scalar;
    }

    inline float2x2 operator*(float scalar, float2x2 mat) noexcept
    {
        return mat *= scalar;
    }

    inline float2x2 operator/(float2x2 mat, float scalar) noexcept
    {
        MATH_ASSERT(scalar != 0.0f);
        return mat /= scalar;
    }

    inline float2 operator*(const float2& vec, const float2x2& mat) noexcept
    {
        return mat.transform_vector_left(vec);
    }

    inline float2 operator*(const float2x2& mat, const float2& vec) noexcept
    {
        return mat.transform_vector(vec);
    }

    // ============================================================================
    // Global Functions
    // ============================================================================

    inline float2x2 transpose(const float2x2& mat) noexcept
    {
        return mat.transposed();
    }

    inline float2x2 inverse(const float2x2& mat) noexcept
    {
        return mat.inverted();
    }

    inline float determinant(const float2x2& mat) noexcept
    {
        return mat.determinant();
    }

    inline float2 mul(const float2& vec, const float2x2& mat) noexcept
    {
        return vec * mat;
    }

    inline float2 mul(const float2x2& mat, const float2& vec) noexcept
    {
        return mat * vec;
    }

    inline float2x2 mul(const float2x2& lhs, const float2x2& rhs) noexcept
    {
        return lhs * rhs;
    }

    inline float trace(const float2x2& mat) noexcept
    {
        return mat.trace();
    }

    inline float2 diagonal(const float2x2& mat) noexcept
    {
        return mat.diagonal();
    }

    inline float frobenius_norm(const float2x2& mat) noexcept
    {
        return mat.frobenius_norm();
    }

    inline bool approximately(const float2x2& a, const float2x2& b, float epsilon) noexcept
    {
        return a.approximately(b, epsilon);
    }

    inline bool is_orthogonal(const float2x2& mat, float epsilon) noexcept
    {
        return mat.is_orthogonal(epsilon);
    }

    inline bool is_rotation(const float2x2& mat, float epsilon) noexcept
    {
        return mat.is_rotation(epsilon);
    }

    inline float2x2 lerp(const float2x2& a, const float2x2& b, float t) noexcept
    {
        return a.lerp(b, t);
    }

    inline float2x2 slerp(const float2x2& a, const float2x2& b, float t) noexcept
    {
        return a.slerp(b, t);
    }

    // ============================================================================
    // Useful Constants
    // ============================================================================

    inline const float2x2 float2x2_Identity = float2x2::identity();
    inline const float2x2 float2x2_Zero = float2x2::zero();
    inline const float2x2 float2x2_ReflectX = float2x2(1.0f, 0.0f, 0.0f, -1.0f);
    inline const float2x2 float2x2_ReflectY = float2x2(-1.0f, 0.0f, 0.0f, 1.0f);

} // namespace Math

#endif // MATH_FLOAT2X2_INL
