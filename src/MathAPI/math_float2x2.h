// Description: 2x2 matrix class with comprehensive mathematical operations,
//              SSE optimization, and full linear algebra support for 2D graphics
//              Column-major order for HLSL compatibility
// Author: NSDeathman, DeepSeek
#pragma once

#include <cmath>
#include <string>
#include <cstdio>
#include <cassert>
#include <xmmintrin.h>
#include <pmmintrin.h>

#include "math_config.h"
#include "math_constants.h"
#include "math_functions.h"
#include "math_float2.h"

namespace Math
{
    /**
     * @class float2x2
     * @brief 2x2 matrix class with comprehensive mathematical operations
     *
     * Represents a 2x2 matrix stored in strict column-major order for HLSL compatibility.
     * Column 0: [m00, m10]
     * Column 1: [m01, m11]
     *
     * Memory layout (column-major):
     * [m00, m10]  // col0
     * [m01, m11]  // col1
     *
     * @note Strict column-major for direct HLSL shader upload
     * @note SSE optimization for performance-critical operations
     * @note Perfect for 2D transformations, linear algebra, and computer vision
     */
    class MATH_API float2x2
    {
    private:
        // Strict column-major storage for HLSL compatibility
        union
        {
            struct
            {
                float m00, m10;  // Column 0
                float m01, m11;  // Column 1
            };
            float data[4];
            __m128 sse;
        };

    public:
        // ============================================================================
        // Constructors
        // ============================================================================

        /**
         * @brief Default constructor (initializes to identity matrix)
         */
        float2x2() noexcept;

        /**
         * @brief Construct from column vectors
         * @param col0 First column vector [m00, m10]
         * @param col1 Second column vector [m01, m11]
         */
        float2x2(const float2& col0, const float2& col1) noexcept;

        /**
         * @brief Construct from components in column-major order
         * @param m00 Column 0, Row 0
         * @param m10 Column 0, Row 1
         * @param m01 Column 1, Row 0
         * @param m11 Column 1, Row 1
         */
        float2x2(float m00, float m10, float m01, float m11) noexcept;

        /**
         * @brief Construct from column-major array
         * @param data Column-major array of 4 elements
         * @note Expected order: [m00, m10, m01, m11] (HLSL compatible)
         */
        explicit float2x2(const float* data) noexcept;

        /**
         * @brief Construct from scalar (diagonal matrix)
         * @param scalar Value for diagonal elements
         */
        explicit float2x2(float scalar) noexcept;

        /**
         * @brief Construct from diagonal vector
         * @param diagonal Diagonal elements [x, y]
         */
        explicit float2x2(const float2& diagonal) noexcept;

        /**
         * @brief Construct from SSE data
         * @param sse_data SSE register containing matrix data in column-major order
         */
        explicit float2x2(__m128 sse_data) noexcept;

        float2x2(const float2x2&) noexcept = default;

        // ============================================================================
        // Assignment Operators
        // ============================================================================

        float2x2& operator=(const float2x2&) noexcept = default;

        // ============================================================================
        // Static Constructors
        // ============================================================================

        /**
         * @brief Identity matrix
         * @return 2x2 identity matrix
         */
        static float2x2 identity() noexcept;

        /**
         * @brief Zero matrix
         * @return 2x2 zero matrix
         */
        static float2x2 zero() noexcept;

        /**
         * @brief Rotation matrix (counter-clockwise)
         * @param angle Rotation angle in radians
         * @return 2D rotation matrix
         * @note Positive angle = counter-clockwise rotation
         */
        static float2x2 rotation(float angle) noexcept;

        /**
         * @brief Scaling matrix
         * @param scale Scale factors [x, y]
         * @return 2D scaling matrix
         */
        static float2x2 scaling(const float2& scale) noexcept;

        /**
         * @brief Scaling matrix from components
         * @param x X scale
         * @param y Y scale
         * @return Scaling matrix
         */
        static float2x2 scaling(float x, float y) noexcept;

        /**
         * @brief Uniform scaling matrix
         * @param uniformScale Uniform scale factor
         * @return Scaling matrix
         */
        static float2x2 scaling(float uniformScale) noexcept;

        /**
         * @brief Shear matrix
         * @param shear Shear factors [x, y]
         * @return 2D shear matrix
         */
        static float2x2 shear(const float2& shear) noexcept;

        /**
         * @brief Shear matrix from components
         * @param x X shear factor
         * @param y Y shear factor
         * @return Shear matrix
         */
        static float2x2 shear(float x, float y) noexcept;

        /**
         * @brief Reflection matrix about a normalized axis
         * @param axis Normalized reflection axis
         * @return Reflection matrix
         */
        static float2x2 reflection(const float2& axis) noexcept;

        /**
         * @brief Create orthonormal basis from X axis
         * @param x_axis X axis (will be normalized)
         * @return Orthonormal basis matrix
         */
        static float2x2 orthonormal_basis_from_x(const float2& x_axis) noexcept;

        /**
         * @brief Create orthonormal basis from Y axis
         * @param y_axis Y axis (will be normalized)
         * @return Orthonormal basis matrix
         */
        static float2x2 orthonormal_basis_from_y(const float2& y_axis) noexcept;

        // ============================================================================
        // Access Operators
        // ============================================================================

        /**
         * @brief Access element by row and column
         * @param row Row index (0 or 1)
         * @param col Column index (0 or 1)
         * @return Reference to element
         */
        float& operator()(int row, int col) noexcept;

        /**
         * @brief Access element by row and column (const)
         * @param row Row index (0 or 1)
         * @param col Column index (0 or 1)
         * @return Const reference to element
         */
        const float& operator()(int row, int col) const noexcept;

        /**
         * @brief Access column by index
         * @param colIndex Column index (0 or 1)
         * @return Column as float2
         */
        float2 col(int colIndex) const noexcept;

        /**
         * @brief Access row by index
         * @param rowIndex Row index (0 or 1)
         * @return Row as float2
         */
        float2 row(int rowIndex) const noexcept;

        // ============================================================================
        // Column and Row Accessors
        // ============================================================================

        float2 col0() const noexcept;
        float2 col1() const noexcept;
        float2 row0() const noexcept;
        float2 row1() const noexcept;
        void set_col0(const float2& col) noexcept;
        void set_col1(const float2& col) noexcept;
        void set_row0(const float2& row) noexcept;
        void set_row1(const float2& row) noexcept;

        // ============================================================================
        // SSE Accessors
        // ============================================================================

        __m128 sse_data() const noexcept;
        void set_sse_data(__m128 sse_data) noexcept;

        // ============================================================================
        // Compound Assignment Operators
        // ============================================================================

        float2x2& operator+=(const float2x2& rhs) noexcept;
        float2x2& operator-=(const float2x2& rhs) noexcept;
        float2x2& operator*=(float scalar) noexcept;
        float2x2& operator/=(float scalar) noexcept;
        float2x2& operator*=(const float2x2& rhs) noexcept;

        // ============================================================================
        // Unary Operators
        // ============================================================================

        float2x2 operator+() const noexcept;
        float2x2 operator-() const noexcept;

        // ============================================================================
        // Matrix Operations
        // ============================================================================

        float2x2 transposed() const noexcept;
        float determinant() const noexcept;
        float2x2 inverted() const noexcept;
        float2x2 adjugate() const noexcept;
        float trace() const noexcept;
        float2 diagonal() const noexcept;
        float frobenius_norm() const noexcept;

        // ============================================================================
        // Vector Transformations
        // ============================================================================

        float2 transform_vector(const float2& vec) const noexcept;
        float2 transform_point(const float2& point) const noexcept;
        float2 transform_vector_left(const float2& vec) const noexcept;

        // ============================================================================
        // Decomposition Methods
        // ============================================================================

        /**
         * @brief Decompose matrix into rotation and scale (without shear)
         * @param rotation Output rotation angle in radians
         * @param scale Output scale vector
         * @return True if decomposition succeeded
         */
        bool decompose_rotation_scale(float& rotation, float2& scale) const noexcept;

        /**
         * @brief Decompose matrix into rotation, scale and shear (if present)
         * @param rotation Output rotation angle in radians
         * @param scale Output scale vector
         * @param shear Output shear vector
         * @return True if decomposition succeeded
         */
        bool decompose_rotation_scale_shear(float& rotation, float2& scale, float2& shear) const noexcept;

        // ============================================================================
        // Transformation Component Extraction
        // ============================================================================

        float get_rotation() const noexcept;
        float2 get_scale() const noexcept;
        float2 get_shear() const noexcept;

        void set_rotation(float angle) noexcept;
        void set_scale(const float2& scale) noexcept;
        void set_scale_preserve_shear(const float2& scale) noexcept;
        void set_shear(const float2& shear) noexcept;

        // ============================================================================
        // Utility Methods
        // ============================================================================

        bool is_identity(float epsilon = Constants::Constants<float>::Epsilon) const noexcept;
        bool is_orthogonal(float epsilon = Constants::Constants<float>::Epsilon) const noexcept;
        bool is_rotation(float epsilon = Constants::Constants<float>::Epsilon) const noexcept;
        bool is_scale_uniform(float epsilon = Constants::Constants<float>::Epsilon) const noexcept;
        bool approximately(const float2x2& other, float epsilon = Constants::Constants<float>::Epsilon) const noexcept;
        bool approximately_zero(float epsilon = Constants::Constants<float>::Epsilon) const noexcept;

        std::string to_string() const;

        const float* column_major_data() const noexcept;
        void to_column_major(float* data) const noexcept;
        void to_row_major(float* data) const noexcept;

        // ============================================================================
        // Comparison Operators
        // ============================================================================

        bool operator==(const float2x2& rhs) const noexcept;
        bool operator!=(const float2x2& rhs) const noexcept;
        bool operator<(const float2x2& rhs) const noexcept;  // For sorting/containers

        // ============================================================================
        // Specialized Operations
        // ============================================================================

        /**
         * @brief Orthonormalize the matrix (Gram-Schmidt process)
         * @param epsilon Tolerance for zero vectors
         * @return Orthonormalized matrix
         */
        float2x2 orthonormalized(float epsilon = Constants::Constants<float>::Epsilon) const noexcept;

        /**
         * @brief Extract closest rotation matrix (polar decomposition)
         * @return Closest rotation matrix
         */
        float2x2 closest_rotation() const noexcept;

        /**
         * @brief Linear interpolation between matrices
         * @param b Target matrix
         * @param t Interpolation factor [0, 1]
         * @return Interpolated matrix
         */
        float2x2 lerp(const float2x2& b, float t) const noexcept;

        /**
         * @brief Spherical linear interpolation for rotation matrices
         * @param b Target rotation matrix
         * @param t Interpolation factor [0, 1]
         * @return Interpolated rotation matrix
         */
        float2x2 slerp(const float2x2& b, float t) const noexcept;
    };

    // ============================================================================
    // Binary Operators
    // ============================================================================

    float2x2 operator+(float2x2 lhs, const float2x2& rhs) noexcept;
    float2x2 operator-(float2x2 lhs, const float2x2& rhs) noexcept;
    float2x2 operator*(const float2x2& lhs, const float2x2& rhs) noexcept;
    float2x2 operator*(float2x2 mat, float scalar) noexcept;
    float2x2 operator*(float scalar, float2x2 mat) noexcept;
    float2x2 operator/(float2x2 mat, float scalar) noexcept;

    float2 operator*(const float2& vec, const float2x2& mat) noexcept;

    // ============================================================================
    // Global Functions
    // ============================================================================

    float2x2 transpose(const float2x2& mat) noexcept;
    float2x2 inverse(const float2x2& mat) noexcept;
    float determinant(const float2x2& mat) noexcept;
    float2 mul(const float2& vec, const float2x2& mat) noexcept;
    float2 mul(const float2x2& mat, const float2& vec) noexcept;
    float2x2 mul(const float2x2& lhs, const float2x2& rhs) noexcept;
    float trace(const float2x2& mat) noexcept;
    float2 diagonal(const float2x2& mat) noexcept;
    float frobenius_norm(const float2x2& mat) noexcept;

    bool approximately(const float2x2& a, const float2x2& b,
        float epsilon = Constants::Constants<float>::Epsilon) noexcept;
    bool is_orthogonal(const float2x2& mat,
        float epsilon = Constants::Constants<float>::Epsilon) noexcept;
    bool is_rotation(const float2x2& mat,
        float epsilon = Constants::Constants<float>::Epsilon) noexcept;

    float2x2 lerp(const float2x2& a, const float2x2& b, float t) noexcept;
    float2x2 slerp(const float2x2& a, const float2x2& b, float t) noexcept;

    // ============================================================================
    // Useful Constants
    // ============================================================================

    extern const float2x2 float2x2_Identity;
    extern const float2x2 float2x2_Zero;
    extern const float2x2 float2x2_ReflectX;
    extern const float2x2 float2x2_ReflectY;

} // namespace Math

// Include inline implementation
#include "math_float2x2.inl"
