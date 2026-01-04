namespace Math
{
    // ============================================================================
    // Constructors Implementation (COLUMN-MAJOR)
    // ============================================================================

    inline float3x3::float3x3() noexcept
        : col0_(1.0f, 0.0f, 0.0f, 0.0f),    // First column: [1, 0, 0]^T
        col1_(0.0f, 1.0f, 0.0f, 0.0f),    // Second column: [0, 1, 0]^T
        col2_(0.0f, 0.0f, 1.0f, 0.0f)     // Third column: [0, 0, 1]^T
    {}

    inline float3x3::float3x3(const float3& col0, const float3& col1, const float3& col2) noexcept
    {
        // Direct column assignment (COLUMN-MAJOR)
        col0_ = float4(col0.x, col0.y, col0.z, 0.0f);
        col1_ = float4(col1.x, col1.y, col1.z, 0.0f);
        col2_ = float4(col2.x, col2.y, col2.z, 0.0f);
    }

    inline float3x3::float3x3(float m00, float m01, float m02,
        float m10, float m11, float m12,
        float m20, float m21, float m22) noexcept
    {
        // Convert ROW-MAJOR parameters to COLUMN-MAJOR storage
        // Row 0: m00, m01, m02 -> becomes: col0.x, col1.x, col2.x
        // Row 1: m10, m11, m12 -> becomes: col0.y, col1.y, col2.y
        // Row 2: m20, m21, m22 -> becomes: col0.z, col1.z, col2.z
        col0_ = float4(m00, m10, m20, 0.0f);  // First column: [m00, m10, m20]^T
        col1_ = float4(m01, m11, m21, 0.0f);  // Second column: [m01, m11, m21]^T
        col2_ = float4(m02, m12, m22, 0.0f);  // Third column: [m02, m12, m22]^T
    }

    inline float3x3::float3x3(const float* data) noexcept
    {
        // Input is ROW-MAJOR: [r0c0, r0c1, r0c2, r1c0, r1c1, r1c2, r2c0, r2c1, r2c2]
        // Convert to COLUMN-MAJOR storage
        col0_ = float4(data[0], data[3], data[6], 0.0f);  // First column: [r0c0, r1c0, r2c0]^T
        col1_ = float4(data[1], data[4], data[7], 0.0f);  // Second column: [r0c1, r1c1, r2c1]^T
        col2_ = float4(data[2], data[5], data[8], 0.0f);  // Third column: [r0c2, r1c2, r2c2]^T
    }

    inline float3x3::float3x3(float scalar) noexcept
        : col0_(scalar, 0, 0, 0),    // First column: [scalar, 0, 0]^T
        col1_(0, scalar, 0, 0),    // Second column: [0, scalar, 0]^T
        col2_(0, 0, scalar, 0)     // Third column: [0, 0, scalar]^T
    {}

    inline float3x3::float3x3(const float3& diagonal) noexcept
        : col0_(diagonal.x, 0, 0, 0),    // First column: [diag.x, 0, 0]^T
        col1_(0, diagonal.y, 0, 0),    // Second column: [0, diag.y, 0]^T
        col2_(0, 0, diagonal.z, 0)     // Third column: [0, 0, diag.z]^T
    {}

    inline float3x3::float3x3(const float4x4& mat4x4) noexcept
    {
        // Extract upper-left 3x3 from 4x4 matrix (both COLUMN-MAJOR)
        const __m128 mask = _mm_set_ps(0.0f, 1.0f, 1.0f, 1.0f);
        col0_ = float4(_mm_and_ps(mat4x4.col0().get_simd(), mask));
        col1_ = float4(_mm_and_ps(mat4x4.col1().get_simd(), mask));
        col2_ = float4(_mm_and_ps(mat4x4.col2().get_simd(), mask));
    }

    inline float3x3::float3x3(const quaternion& q) noexcept
    {
        // Generate rotation matrix from quaternion (COLUMN-MAJOR)
        const float xx = q.x * q.x;
        const float yy = q.y * q.y;
        const float zz = q.z * q.z;
        const float xy = q.x * q.y;
        const float xz = q.x * q.z;
        const float yz = q.y * q.z;
        const float wx = q.w * q.x;
        const float wy = q.w * q.y;
        const float wz = q.w * q.z;

        // First column: [1 - 2(yy + zz), 2(xy + wz), 2(xz - wy)]^T
        col0_ = float4(1.0f - 2.0f * (yy + zz), 2.0f * (xy + wz), 2.0f * (xz - wy), 0.0f);

        // Second column: [2(xy - wz), 1 - 2(xx + zz), 2(yz + wx)]^T
        col1_ = float4(2.0f * (xy - wz), 1.0f - 2.0f * (xx + zz), 2.0f * (yz + wx), 0.0f);

        // Third column: [2(xz + wy), 2(yz - wx), 1 - 2(xx + yy)]^T
        col2_ = float4(2.0f * (xz + wy), 2.0f * (yz - wx), 1.0f - 2.0f * (xx + yy), 0.0f);
    }

    // ============================================================================
    // Assignment Operators Implementation
    // ============================================================================

    inline float3x3& float3x3::operator=(const float4x4& mat4x4) noexcept
    {
        // Extract upper-left 3x3 from 4x4 matrix (both COLUMN-MAJOR)
        const __m128 mask = _mm_set_ps(0.0f, 1.0f, 1.0f, 1.0f);
        col0_ = float4(_mm_and_ps(mat4x4.col0().get_simd(), mask));
        col1_ = float4(_mm_and_ps(mat4x4.col1().get_simd(), mask));
        col2_ = float4(_mm_and_ps(mat4x4.col2().get_simd(), mask));
        return *this;
    }

    // ============================================================================
    // Access Operators Implementation (COLUMN-MAJOR)
    // ============================================================================

    inline float3& float3x3::operator[](int colIndex) noexcept
    {
        // COLUMN-MAJOR: index refers to column number
        switch (colIndex) {
        case 0: return *reinterpret_cast<float3*>(&col0_);
        case 1: return *reinterpret_cast<float3*>(&col1_);
        case 2: return *reinterpret_cast<float3*>(&col2_);
        default:
            static float3 dummy;
            return dummy;
        }
    }

    inline const float3& float3x3::operator[](int colIndex) const noexcept
    {
        // COLUMN-MAJOR: index refers to column number
        switch (colIndex) {
        case 0: return *reinterpret_cast<const float3*>(&col0_);
        case 1: return *reinterpret_cast<const float3*>(&col1_);
        case 2: return *reinterpret_cast<const float3*>(&col2_);
        default:
            static float3 dummy;
            return dummy;
        }
    }

    inline float& float3x3::operator()(int row, int col) noexcept
    {
        // COLUMN-MAJOR: m[col][row]
        switch (col) {
        case 0:
            switch (row) {
            case 0: return col0_.x;
            case 1: return col0_.y;
            case 2: return col0_.z;
            }
        case 1:
            switch (row) {
            case 0: return col1_.x;
            case 1: return col1_.y;
            case 2: return col1_.z;
            }
        case 2:
            switch (row) {
            case 0: return col2_.x;
            case 1: return col2_.y;
            case 2: return col2_.z;
            }
        }
        return col0_.x;
    }

    inline const float& float3x3::operator()(int row, int col) const noexcept
    {
        // COLUMN-MAJOR: m[col][row]
        switch (col) {
        case 0:
            switch (row) {
            case 0: return col0_.x;
            case 1: return col0_.y;
            case 2: return col0_.z;
            }
        case 1:
            switch (row) {
            case 0: return col1_.x;
            case 1: return col1_.y;
            case 2: return col1_.z;
            }
        case 2:
            switch (row) {
            case 0: return col2_.x;
            case 1: return col2_.y;
            case 2: return col2_.z;
            }
        }
        return col0_.x;
    }

    // ============================================================================
    // Column and Row Accessors Implementation
    // ============================================================================

    inline float3 float3x3::col0() const noexcept { return float3(col0_.x, col0_.y, col0_.z); }
    inline float3 float3x3::col1() const noexcept { return float3(col1_.x, col1_.y, col1_.z); }
    inline float3 float3x3::col2() const noexcept { return float3(col2_.x, col2_.y, col2_.z); }

    inline float3 float3x3::row0() const noexcept { return float3(col0_.x, col1_.x, col2_.x); }
    inline float3 float3x3::row1() const noexcept { return float3(col0_.y, col1_.y, col2_.y); }
    inline float3 float3x3::row2() const noexcept { return float3(col0_.z, col1_.z, col2_.z); }

    inline void float3x3::set_col0(const float3& col) noexcept
    {
        col0_.x = col.x;
        col0_.y = col.y;
        col0_.z = col.z;
    }

    inline void float3x3::set_col1(const float3& col) noexcept
    {
        col1_.x = col.x;
        col1_.y = col.y;
        col1_.z = col.z;
    }

    inline void float3x3::set_col2(const float3& col) noexcept
    {
        col2_.x = col.x;
        col2_.y = col.y;
        col2_.z = col.z;
    }

    inline void float3x3::set_row0(const float3& row) noexcept
    {
        col0_.x = row.x;  // Row 0, Column 0
        col1_.x = row.y;  // Row 0, Column 1
        col2_.x = row.z;  // Row 0, Column 2
    }

    inline void float3x3::set_row1(const float3& row) noexcept
    {
        col0_.y = row.x;  // Row 1, Column 0
        col1_.y = row.y;  // Row 1, Column 1
        col2_.y = row.z;  // Row 1, Column 2
    }

    inline void float3x3::set_row2(const float3& row) noexcept
    {
        col0_.z = row.x;  // Row 2, Column 0
        col1_.z = row.y;  // Row 2, Column 1
        col2_.z = row.z;  // Row 2, Column 2
    }

    // ============================================================================
    // Static Constructors Implementation (COLUMN-MAJOR)
    // ============================================================================

    inline float3x3 float3x3::identity() noexcept
    {
        return float3x3(float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1));
    }

    inline float3x3 float3x3::zero() noexcept
    {
        return float3x3(float3(0, 0, 0), float3(0, 0, 0), float3(0, 0, 0));
    }

    inline bool float3x3::approximately_zero(float epsilon) const noexcept
    {
        return col0_.approximately_zero(epsilon) &&
            col1_.approximately_zero(epsilon) &&
            col2_.approximately_zero(epsilon);
    }

    inline float3x3 float3x3::scaling(const float3& scale) noexcept
    {
        return float3x3(float3(scale.x, 0, 0), float3(0, scale.y, 0), float3(0, 0, scale.z));
    }

    inline float3x3 float3x3::scaling(const float scaleX, const float scaleY, const float scaleZ) noexcept
    {
        return float3x3(float3(scaleX, 0, 0), float3(0, scaleY, 0), float3(0, 0, scaleZ));
    }

    inline float3x3 float3x3::scaling(float scale) noexcept
    {
        return float3x3(scale);
    }

    inline float3x3 float3x3::rotation_x(float angle) noexcept
    {
        float s, c;
        MathFunctions::sin_cos(angle, &s, &c);

        // COLUMN-MAJOR: each float3 is a column
        return float3x3(
            float3(1.0f, 0.0f, 0.0f),  // First column
            float3(0.0f, c, s),        // Second column
            float3(0.0f, -s, c)        // Third column
        );
    }

    inline float3x3 float3x3::rotation_y(float angle) noexcept
    {
        float s, c;
        MathFunctions::sin_cos(angle, &s, &c);

        // COLUMN-MAJOR: each float3 is a column
        return float3x3(
            float3(c, 0.0f, -s),       // First column
            float3(0.0f, 1.0f, 0.0f),  // Second column
            float3(s, 0.0f, c)         // Third column
        );
    }

    inline float3x3 float3x3::rotation_z(float angle) noexcept
    {
        float s, c;
        MathFunctions::sin_cos(angle, &s, &c);

        // COLUMN-MAJOR: each float3 is a column
        return float3x3(
            float3(c, s, 0.0f),        // First column
            float3(-s, c, 0.0f),       // Second column
            float3(0.0f, 0.0f, 1.0f)   // Third column
        );
    }

    inline float3x3 float3x3::rotation_axis(const float3& axis, float angle) noexcept
    {
        if (axis.approximately_zero(Constants::Constants<float>::Epsilon)) {
            return identity();
        }

        float s, c;
        MathFunctions::sin_cos(angle, &s, &c);

        const float one_minus_c = 1.0f - c;
        const float3 n = axis.normalize();

        const float x = n.x, y = n.y, z = n.z;
        const float xx = x * x, yy = y * y, zz = z * z;
        const float xy = x * y, xz = x * z, yz = y * z;
        const float xs = x * s, ys = y * s, zs = z * s;

        // COLUMN-MAJOR construction using Rodrigues' rotation formula
        __m128 col0 = _mm_set_ps(0.0f, xz * one_minus_c - ys, xy * one_minus_c + zs, xx * one_minus_c + c);
        __m128 col1 = _mm_set_ps(0.0f, yz * one_minus_c + xs, yy * one_minus_c + c, xy * one_minus_c - zs);
        __m128 col2 = _mm_set_ps(0.0f, zz * one_minus_c + c, yz * one_minus_c - xs, xz * one_minus_c + ys);

        float3x3 result;
        _mm_store_ps(&result.col0_.x, col0);
        _mm_store_ps(&result.col1_.x, col1);
        _mm_store_ps(&result.col2_.x, col2);

        return result;
    }

    inline float3x3 float3x3::rotation_euler(const float3& angles) noexcept
    {
        // For COLUMN-MAJOR: M = Mz * My * Mx (applied from right to left)
        return rotation_z(angles.z) * rotation_y(angles.y) * rotation_x(angles.x);
    }

    inline float3x3 float3x3::skew_symmetric(const float3& vec) noexcept
    {
        // COLUMN-MAJOR: each column is cross(vec, basis_vector)
        return float3x3(
            float3(0, vec.z, -vec.y),      // cross(vec, [1,0,0])
            float3(-vec.z, 0, vec.x),      // cross(vec, [0,1,0])
            float3(vec.y, -vec.x, 0)       // cross(vec, [0,0,1])
        );
    }

    inline float3x3 float3x3::outer_product(const float3& u, const float3& v) noexcept
    {
        // Outer product u * v^T (COLUMN-MAJOR)
        // Column i = u * v[i]
        return float3x3(
            float3(u.x * v.x, u.y * v.x, u.z * v.x),  // First column: u * v.x
            float3(u.x * v.y, u.y * v.y, u.z * v.y),  // Second column: u * v.y
            float3(u.x * v.z, u.y * v.z, u.z * v.z)   // Third column: u * v.z
        );
    }

    // ============================================================================
    // Compound Assignment Operators Implementation
    // ============================================================================

    inline float3x3& float3x3::operator+=(const float3x3& rhs) noexcept
    {
        col0_ += rhs.col0_;
        col1_ += rhs.col1_;
        col2_ += rhs.col2_;
        return *this;
    }

    inline float3x3& float3x3::operator-=(const float3x3& rhs) noexcept
    {
        col0_ -= rhs.col0_;
        col1_ -= rhs.col1_;
        col2_ -= rhs.col2_;
        return *this;
    }

    inline float3x3& float3x3::operator*=(float scalar) noexcept
    {
        col0_ *= scalar;
        col1_ *= scalar;
        col2_ *= scalar;
        return *this;
    }

    inline float3x3& float3x3::operator/=(float scalar) noexcept
    {
        const float inv_scalar = 1.0f / scalar;
        col0_ *= inv_scalar;
        col1_ *= inv_scalar;
        col2_ *= inv_scalar;
        return *this;
    }

    inline float3x3& float3x3::operator*=(const float3x3& rhs) noexcept
    {
        *this = *this * rhs;
        return *this;
    }

    // ============================================================================
    // Unary Operators Implementation
    // ============================================================================

    inline float3x3 float3x3::operator+() const noexcept { return *this; }

    inline float3x3 float3x3::operator-() const noexcept
    {
        return float3x3(
            -col0_.xyz(),
            -col1_.xyz(),
            -col2_.xyz()
        );
    }

    // ============================================================================
    // Matrix Operations Implementation (COLUMN-MAJOR)
    // ============================================================================

    inline float3x3 float3x3::transposed() const noexcept
    {
        // Transpose by extracting rows and converting them to columns
        return float3x3(
            row0(),  // First row becomes first column
            row1(),  // Second row becomes second column
            row2()   // Third row becomes third column
        );
    }

    inline float float3x3::determinant() const noexcept
    {
        const float3 col0 = col0_.xyz();
        const float3 col1 = col1_.xyz();
        const float3 col2 = col2_.xyz();

        // Determinant formula for COLUMN-MAJOR: det = col0·(col1×col2)
        const float a = col0.x, b = col0.y, c = col0.z;
        const float d = col1.x, e = col1.y, f = col1.z;
        const float g = col2.x, h = col2.y, i = col2.z;

        return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    }

    inline float3x3 float3x3::inverted() const noexcept
    {
        __m128 c0 = _mm_load_ps(&col0_.x);
        __m128 c1 = _mm_load_ps(&col1_.x);
        __m128 c2 = _mm_load_ps(&col2_.x);

        // Compute determinant using SSE
        __m128 c1_yzx = _mm_shuffle_ps(c1, c1, _MM_SHUFFLE(3, 0, 2, 1));
        __m128 c2_yzx = _mm_shuffle_ps(c2, c2, _MM_SHUFFLE(3, 0, 2, 1));
        __m128 c1_zxy = _mm_shuffle_ps(c1, c1, _MM_SHUFFLE(3, 1, 0, 2));
        __m128 c2_zxy = _mm_shuffle_ps(c2, c2, _MM_SHUFFLE(3, 1, 0, 2));

        __m128 minor_products = _mm_sub_ps(_mm_mul_ps(c1_yzx, c2_zxy),
            _mm_mul_ps(c1_zxy, c2_yzx));

        __m128 det_terms = _mm_mul_ps(c0, minor_products);
        __m128 signed_terms = _mm_mul_ps(det_terms, _mm_set_ps(0.0f, 1.0f, -1.0f, 1.0f));

        __m128 det_sum = _mm_hadd_ps(signed_terms, signed_terms);
        det_sum = _mm_hadd_ps(det_sum, det_sum);
        float det = _mm_cvtss_f32(det_sum);

        if (std::abs(det) < Constants::Constants<float>::Epsilon) {
            return identity();
        }

        const float inv_det = 1.0f / det;
        const __m128 inv_det_vec = _mm_set1_ps(inv_det);

        const float a = col0_.x, b = col0_.y, c = col0_.z;
        const float d = col1_.x, e = col1_.y, f = col1_.z;
        const float g = col2_.x, h = col2_.y, i = col2_.z;

        // Compute adjugate matrix (transpose of cofactor matrix)
        // For COLUMN-MAJOR: inverse = (1/det) * adjugate
        __m128 adj00_adj01 = _mm_set_ps(0.0f,
            b * f - c * e,        // cofactor(0,0)
            -(b * i - c * h),     // cofactor(0,1)
            e * i - f * h);       // cofactor(0,2)

        __m128 adj10_adj11 = _mm_set_ps(0.0f,
            -(a * f - c * d),     // cofactor(1,0)
            a * i - c * g,        // cofactor(1,1)
            -(d * i - f * g));    // cofactor(1,2)

        __m128 adj20_adj21 = _mm_set_ps(0.0f,
            a * e - b * d,        // cofactor(2,0)
            -(a * h - b * g),     // cofactor(2,1)
            d * h - e * g);       // cofactor(2,2)

        // Multiply by inverse determinant
        adj00_adj01 = _mm_mul_ps(adj00_adj01, inv_det_vec);
        adj10_adj11 = _mm_mul_ps(adj10_adj11, inv_det_vec);
        adj20_adj21 = _mm_mul_ps(adj20_adj21, inv_det_vec);

        // Construct columns from adjugate rows (transpose)
        __m128 result_col0 = _mm_set_ps(0.0f,
            _mm_cvtss_f32(_mm_shuffle_ps(adj20_adj21, adj20_adj21, _MM_SHUFFLE(0, 0, 0, 0))),  // cofactor(2,0)
            _mm_cvtss_f32(_mm_shuffle_ps(adj10_adj11, adj10_adj11, _MM_SHUFFLE(0, 0, 0, 0))),  // cofactor(1,0)
            _mm_cvtss_f32(adj00_adj01));  // cofactor(0,0)

        __m128 result_col1 = _mm_set_ps(0.0f,
            _mm_cvtss_f32(_mm_shuffle_ps(adj20_adj21, adj20_adj21, _MM_SHUFFLE(1, 1, 1, 1))),  // cofactor(2,1)
            _mm_cvtss_f32(_mm_shuffle_ps(adj10_adj11, adj10_adj11, _MM_SHUFFLE(1, 1, 1, 1))),  // cofactor(1,1)
            _mm_cvtss_f32(_mm_shuffle_ps(adj00_adj01, adj00_adj01, _MM_SHUFFLE(1, 1, 1, 1)))); // cofactor(0,1)

        __m128 result_col2 = _mm_set_ps(0.0f,
            _mm_cvtss_f32(_mm_shuffle_ps(adj20_adj21, adj20_adj21, _MM_SHUFFLE(2, 2, 2, 2))),  // cofactor(2,2)
            _mm_cvtss_f32(_mm_shuffle_ps(adj10_adj11, adj10_adj11, _MM_SHUFFLE(2, 2, 2, 2))),  // cofactor(1,2)
            _mm_cvtss_f32(_mm_shuffle_ps(adj00_adj01, adj00_adj01, _MM_SHUFFLE(2, 2, 2, 2)))); // cofactor(0,2)

        float3x3 result;
        _mm_store_ps(&result.col0_.x, result_col0);
        _mm_store_ps(&result.col1_.x, result_col1);
        _mm_store_ps(&result.col2_.x, result_col2);

        return result;
    }

    inline float3x3 float3x3::normal_matrix(const float3x3& model) noexcept
    {
        float3x3 inv = model.inverted();
        float3x3 result = inv.transposed();

        // Normalize columns for numerical stability
        float3 col0 = result.col0().normalize();
        float3 col1 = result.col1().normalize();
        float3 col2 = result.col2().normalize();

        return float3x3(col0, col1, col2);
    }

    inline float float3x3::trace() const noexcept
    {
        // Trace = sum of diagonal elements
        return col0_.x + col1_.y + col2_.z;
    }

    inline float3 float3x3::diagonal() const noexcept
    {
        // Extract diagonal elements from columns
        return float3(col0_.x, col1_.y, col2_.z);
    }

    inline float float3x3::frobenius_norm() const noexcept
    {
        return std::sqrt(col0_.length_sq() + col1_.length_sq() + col2_.length_sq());
    }

    inline float3x3 float3x3::symmetric_part() const noexcept
    {
        float3x3 trans = transposed();
        return (*this + trans) * 0.5f;
    }

    inline float3x3 float3x3::skew_symmetric_part() const noexcept
    {
        float3x3 trans = transposed();
        return (*this - trans) * 0.5f;
    }

    // ============================================================================
    // Vector Transformations Implementation (COLUMN-MAJOR)
    // ============================================================================

    inline float3 float3x3::transform_vector(const float3& vec) const noexcept
    {
        // COLUMN-MAJOR: result = col0*vec.x + col1*vec.y + col2*vec.z
        __m128 v = _mm_set_ps(0.0f, vec.z, vec.y, vec.x);
        __m128 vx = _mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0));
        __m128 vy = _mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1));
        __m128 vz = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2));

        __m128 col0 = _mm_load_ps(&col0_.x);
        __m128 col1 = _mm_load_ps(&col1_.x);
        __m128 col2 = _mm_load_ps(&col2_.x);

        __m128 result = _mm_mul_ps(col0, vx);
        result = _mm_add_ps(result, _mm_mul_ps(col1, vy));
        result = _mm_add_ps(result, _mm_mul_ps(col2, vz));

        return float3(result);
    }

    inline float3 float3x3::transform_point(const float3& point) const noexcept
    {
        // For 3x3 matrix, same as vector transformation
        return transform_vector(point);
    }

    inline float3 float3x3::transform_normal(const float3& normal) const noexcept
    {
        // Transform normal with transpose(inverse(matrix))
        // Equivalent to: (inverse(matrix))^T * normal
        float3x3 inv = this->inverted();
        float3x3 normalMat = inv.transposed();
        return normalMat.transform_vector(normal);
    }

    // ============================================================================
    // Decomposition Methods Implementation (COLUMN-MAJOR aware)
    // ============================================================================

    inline float3 float3x3::extract_scale() const noexcept
    {
        // Scale factors are lengths of column vectors
        return float3(col0_.length(), col1_.length(), col2_.length());
    }

    inline float3x3 float3x3::extract_rotation() const noexcept
    {
        // Gram-Schmidt orthogonalization on columns
        if (is_orthonormal(1e-4f)) {
            return *this;
        }

        __m128 c0 = _mm_load_ps(&col0_.x);
        __m128 c1 = _mm_load_ps(&col1_.x);
        __m128 c2 = _mm_load_ps(&col2_.x);

        // Normalize first column
        __m128 len0 = _mm_sqrt_ps(_mm_dp_ps(c0, c0, 0x7F));
        c0 = _mm_div_ps(c0, len0);

        // Orthogonalize second column against first
        __m128 dot01 = _mm_dp_ps(c1, c0, 0x7F);
        c1 = _mm_sub_ps(c1, _mm_mul_ps(dot01, c0));
        __m128 len1 = _mm_sqrt_ps(_mm_dp_ps(c1, c1, 0x7F));
        c1 = _mm_div_ps(c1, len1);

        // Third column as cross product of first two (ensures right-handed)
        __m128 c0_yzx = _mm_shuffle_ps(c0, c0, _MM_SHUFFLE(3, 0, 2, 1));
        __m128 c1_yzx = _mm_shuffle_ps(c1, c1, _MM_SHUFFLE(3, 0, 2, 1));
        __m128 c0_zxy = _mm_shuffle_ps(c0, c0, _MM_SHUFFLE(3, 1, 0, 2));
        __m128 c1_zxy = _mm_shuffle_ps(c1, c1, _MM_SHUFFLE(3, 1, 0, 2));

        c2 = _mm_sub_ps(_mm_mul_ps(c0_yzx, c1_zxy), _mm_mul_ps(c0_zxy, c1_yzx));

        float3x3 result;
        _mm_store_ps(&result.col0_.x, c0);
        _mm_store_ps(&result.col1_.x, c1);
        _mm_store_ps(&result.col2_.x, c2);

        return result;
    }

    // ============================================================================
    // Utility Methods Implementation
    // ============================================================================

    inline bool float3x3::is_identity(float epsilon) const noexcept
    {
        return col0_.approximately(float4(1, 0, 0, 0), epsilon) &&
            col1_.approximately(float4(0, 1, 0, 0), epsilon) &&
            col2_.approximately(float4(0, 0, 1, 0), epsilon);
    }

    inline bool float3x3::is_orthogonal(float epsilon) const noexcept
    {
        // Check if columns are mutually orthogonal
        return MathFunctions::approximately(dot(col0(), col1()), 0.0f, epsilon) &&
            MathFunctions::approximately(dot(col0(), col2()), 0.0f, epsilon) &&
            MathFunctions::approximately(dot(col1(), col2()), 0.0f, epsilon);
    }

    inline bool float3x3::is_orthonormal(float epsilon) const noexcept
    {
        return is_orthogonal(epsilon) &&
            MathFunctions::approximately(col0().length_sq(), 1.0f, epsilon) &&
            MathFunctions::approximately(col1().length_sq(), 1.0f, epsilon) &&
            MathFunctions::approximately(col2().length_sq(), 1.0f, epsilon);
    }

    inline bool float3x3::approximately(const float3x3& other, float epsilon) const noexcept
    {
        return
            MathFunctions::approximately(col0_.x, other.col0_.x, epsilon) &&
            MathFunctions::approximately(col0_.y, other.col0_.y, epsilon) &&
            MathFunctions::approximately(col0_.z, other.col0_.z, epsilon) &&
            MathFunctions::approximately(col1_.x, other.col1_.x, epsilon) &&
            MathFunctions::approximately(col1_.y, other.col1_.y, epsilon) &&
            MathFunctions::approximately(col1_.z, other.col1_.z, epsilon) &&
            MathFunctions::approximately(col2_.x, other.col2_.x, epsilon) &&
            MathFunctions::approximately(col2_.y, other.col2_.y, epsilon) &&
            MathFunctions::approximately(col2_.z, other.col2_.z, epsilon);
    }

    inline std::string float3x3::to_string() const
    {
        char buffer[256];
        std::snprintf(buffer, sizeof(buffer),
            "[%8.4f, %8.4f, %8.4f]\n"
            "[%8.4f, %8.4f, %8.4f]\n"
            "[%8.4f, %8.4f, %8.4f]",
            (*this)(0, 0), (*this)(0, 1), (*this)(0, 2),  // Row 0
            (*this)(1, 0), (*this)(1, 1), (*this)(1, 2),  // Row 1
            (*this)(2, 0), (*this)(2, 1), (*this)(2, 2)); // Row 2
        return std::string(buffer);
    }

    inline void float3x3::to_column_major(float* data) const noexcept
    {
        // Store in COLUMN-MAJOR order: col0, col1, col2
        data[0] = col0_.x; data[1] = col0_.y; data[2] = col0_.z;  // Column 0
        data[3] = col1_.x; data[4] = col1_.y; data[5] = col1_.z;  // Column 1
        data[6] = col2_.x; data[7] = col2_.y; data[8] = col2_.z;  // Column 2
    }

    inline void float3x3::to_row_major(float* data) const noexcept
    {
        // Store in ROW-MAJOR order: row0, row1, row2
        data[0] = col0_.x; data[1] = col1_.x; data[2] = col2_.x;  // Row 0
        data[3] = col0_.y; data[4] = col1_.y; data[5] = col2_.y;  // Row 1
        data[6] = col0_.z; data[7] = col1_.z; data[8] = col2_.z;  // Row 2
    }

    inline bool float3x3::isValid() const noexcept
    {
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                if (!std::isfinite((*this)(i, j)))
                    return false;
        return true;
    }

    // ============================================================================
    // Comparison Operators Implementation
    // ============================================================================

    inline bool float3x3::operator==(const float3x3& rhs) const noexcept
    {
        return approximately(rhs);
    }

    inline bool float3x3::operator!=(const float3x3& rhs) const noexcept
    {
        return !(*this == rhs);
    }

    // ============================================================================
    // Binary Operators Implementation (Global) - COLUMN-MAJOR aware
    // ============================================================================

    inline float3x3 operator+(float3x3 lhs, const float3x3& rhs) noexcept { return lhs += rhs; }
    inline float3x3 operator-(float3x3 lhs, const float3x3& rhs) noexcept { return lhs -= rhs; }

    inline float3x3 operator*(const float3x3& lhs, const float3x3& rhs) noexcept
    {
        // Standard matrix multiplication for COLUMN-MAJOR matrices
        // Result[i][j] = sum_k lhs[i][k] * rhs[k][j]
        float3x3 result;

        const float* l = &lhs.col0_.x;
        const float* r = &rhs.col0_.x;
        float* res = &result.col0_.x;

        for (int j = 0; j < 3; ++j) {
            __m128 sum = _mm_setzero_ps();

            // Multiply lhs columns by rhs column j elements
            __m128 rhs_val0 = _mm_set1_ps(r[j * 4 + 0]);
            __m128 lhs_col0 = _mm_load_ps(l + 0 * 4);
            sum = _mm_add_ps(sum, _mm_mul_ps(rhs_val0, lhs_col0));

            __m128 rhs_val1 = _mm_set1_ps(r[j * 4 + 1]);
            __m128 lhs_col1 = _mm_load_ps(l + 1 * 4);
            sum = _mm_add_ps(sum, _mm_mul_ps(rhs_val1, lhs_col1));

            __m128 rhs_val2 = _mm_set1_ps(r[j * 4 + 2]);
            __m128 lhs_col2 = _mm_load_ps(l + 2 * 4);
            sum = _mm_add_ps(sum, _mm_mul_ps(rhs_val2, lhs_col2));

            _mm_store_ps(res + j * 4, sum);
        }

        return result;
    }

    inline float3x3 operator*(float3x3 mat, float scalar) noexcept { return mat *= scalar; }
    inline float3x3 operator*(float scalar, float3x3 mat) noexcept { return mat *= scalar; }
    inline float3x3 operator/(float3x3 mat, float scalar) noexcept { return mat /= scalar; }

    inline float3 operator*(const float3& vec, const float3x3& mat) noexcept
    {
        // Vector-matrix multiplication (vector as row vector)
        // result.x = dot(vec, mat.col0), result.y = dot(vec, mat.col1), result.z = dot(vec, mat.col2)
        return float3(
            dot(vec, mat.col0()),
            dot(vec, mat.col1()),
            dot(vec, mat.col2())
        );
    }

    inline float3 operator*(const float3x3& mat, const float3& vec) noexcept
    {
        // Matrix-vector multiplication (vector as column vector)
        // Standard COLUMN-MAJOR transformation
        return mat.transform_vector(vec);
    }

    // ============================================================================
    // Global Functions Implementation
    // ============================================================================

    inline float3x3 transpose(const float3x3& mat) noexcept { return mat.transposed(); }
    inline float3x3 inverse(const float3x3& mat) noexcept { return mat.inverted(); }
    inline float determinant(const float3x3& mat) noexcept { return mat.determinant(); }

    inline float3 mul(const float3& vec, const float3x3& mat) noexcept { return vec * mat; }
    inline float3x3 mul(const float3x3& lhs, const float3x3& rhs) noexcept { return lhs * rhs; }

    inline float trace(const float3x3& mat) noexcept { return mat.trace(); }
    inline float3 diagonal(const float3x3& mat) noexcept { return mat.diagonal(); }
    inline float frobenius_norm(const float3x3& mat) noexcept { return mat.frobenius_norm(); }

    inline bool approximately(const float3x3& a, const float3x3& b, float epsilon) noexcept
    {
        return a.approximately(b, epsilon);
    }

    inline bool is_orthogonal(const float3x3& mat, float epsilon) noexcept { return mat.is_orthogonal(epsilon); }
    inline bool is_orthonormal(const float3x3& mat, float epsilon) noexcept { return mat.is_orthonormal(epsilon); }

    inline float3x3 normal_matrix(const float3x3& model) noexcept
    {
        return float3x3::normal_matrix(model);
    }

    inline float3 extract_scale(const float3x3& mat) noexcept { return mat.extract_scale(); }
    inline float3x3 extract_rotation(const float3x3& mat) noexcept { return mat.extract_rotation(); }
    inline float3x3 skew_symmetric(const float3& vec) noexcept { return float3x3::skew_symmetric(vec); }
    inline float3x3 outer_product(const float3& u, const float3& v) noexcept { return float3x3::outer_product(u, v); }

    // ============================================================================
    // Global Constants Implementation
    // ============================================================================

    inline const float3x3 float3x3_Identity = float3x3::identity();
    inline const float3x3 float3x3_Zero = float3x3::zero();
} // namespace Math
