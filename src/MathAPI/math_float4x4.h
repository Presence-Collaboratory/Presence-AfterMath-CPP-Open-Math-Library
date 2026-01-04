// Description: 4x4 matrix class with comprehensive mathematical operations,
//              SSE optimization, and Column-Major layout for OpenGL/Vulkan compatibility.
// Author: NSDeathMan, DeepSeek
#pragma once

#include <cmath>
#include <string>
#include <cstdio>
#include <algorithm>
#include <xmmintrin.h>
#include <pmmintrin.h>

#include "math_config.h"
#include "math_constants.h"
#include "math_functions.h"
#include "math_float3.h"
#include "math_float4.h"

namespace Math
{
    class float3x3;
    class quaternion;

    /**
     * @class float4x4
     * @brief 4x4 matrix class stored in Column-Major order.
     *
     * Represents a 4x4 matrix stored in column-major order for OpenGL/Vulkan compatibility.
     * Provides comprehensive linear algebra operations including matrix multiplication,
     * inversion, determinant calculation, and various 3D transformation matrices.
     *
     * @note Column-major storage for compatibility with OpenGL, Vulkan and GLSL
     * @note Full SSE optimization for performance-critical operations
     * @note Perfect for 3D transformations, view/projection matrices, and linear algebra
     */
    class MATH_API float4x4
    {
    public:
        // Store matrix as four float4 columns for alignment and SSE optimization
        alignas(16) float4 col0_;  ///< First column (x axis / right vector)
        alignas(16) float4 col1_;  ///< Second column (y axis / up vector)
        alignas(16) float4 col2_;  ///< Third column (z axis / forward vector)
        alignas(16) float4 col3_;  ///< Fourth column (translation / perspective)

    public:
        // ============================================================================
        // Constructors
        // ============================================================================

        /**
         * @brief Default constructor (initializes to identity matrix)
         */
        float4x4() noexcept;

        /**
         * @brief Construct from column vectors
         * @param c0 First column vector
         * @param c1 Second column vector
         * @param c2 Third column vector
         * @param c3 Fourth column vector (translation/perspective)
         */
        float4x4(const float4& c0, const float4& c1, const float4& c2, const float4& c3) noexcept;

        /**
         * @brief Construct from 16 scalar values (column-major order)
         * @param m00 Element at column 0, row 0
         * @param m10 Element at column 0, row 1
         * @param m20 Element at column 0, row 2
         * @param m30 Element at column 0, row 3
         * @param m01 Element at column 1, row 0
         * @param m11 Element at column 1, row 1
         * @param m21 Element at column 1, row 2
         * @param m31 Element at column 1, row 3
         * @param m02 Element at column 2, row 0
         * @param m12 Element at column 2, row 1
         * @param m22 Element at column 2, row 2
         * @param m32 Element at column 2, row 3
         * @param m03 Element at column 3, row 0
         * @param m13 Element at column 3, row 1
         * @param m23 Element at column 3, row 2
         * @param m33 Element at column 3, row 3
         * @note Parameters are in column-major order and stored internally in column-major
         */
        float4x4(float m00, float m10, float m20, float m30,
            float m01, float m11, float m21, float m31,
            float m02, float m12, float m22, float m32,
            float m03, float m13, float m23, float m33) noexcept;

        /**
         * @brief Construct from column-major array
         * @param data Column-major array of 16 elements
         * @note Expected order: [col0.x, col0.y, col0.z, col0.w, col1.x, ...]
         */
        explicit float4x4(const float* data) noexcept;

        /**
         * @brief Construct from scalar (diagonal matrix)
         * @param scalar Value for diagonal elements
         */
        explicit float4x4(float scalar) noexcept;

        /**
         * @brief Construct from diagonal vector
         * @param diagonal Diagonal elements (x, y, z, w)
         */
        explicit float4x4(const float4& diagonal) noexcept;

        /**
         * @brief Construct from 3x3 matrix (extends to 4x4 with identity)
         * @param mat3x3 3x3 matrix to extend
         * @note Adds identity for translation and perspective components
         */
        explicit float4x4(const float3x3& mat3x3) noexcept;

        /**
         * @brief Construct from quaternion (rotation matrix)
         * @param q Unit quaternion representing rotation
         * @note Creates homogeneous rotation matrix with no translation
         * @note Uses SSE optimization for efficient conversion
         */
        explicit float4x4(const quaternion& q) noexcept;

        // ============================================================================
        // Static Constructors
        // ============================================================================

        /**
         * @brief Identity matrix
         * @return 4x4 identity matrix
         */
        static float4x4 identity() noexcept;

        /**
         * @brief Zero matrix
         * @return 4x4 zero matrix
         */
        static float4x4 zero() noexcept;

        // --- Transformations (Column-Major specific) ---

        /**
         * @brief Translation matrix from components
         * @param x X translation
         * @param y Y translation
         * @param z Z translation
         * @return Translation matrix
         */
        static float4x4 translation(float x, float y, float z) noexcept;

        /**
         * @brief Translation matrix
         * @param translation Translation vector
         * @return Translation matrix
         * @note Creates matrix that translates points by the specified vector
         */
        static float4x4 translation(const float3& translation) noexcept;

        /**
         * @brief Scaling matrix from components
         * @param x X scale
         * @param y Y scale
         * @param z Z scale
         * @return Scaling matrix
         */
        static float4x4 scaling(float x, float y, float z) noexcept;

        /**
         * @brief Scaling matrix
         * @param scale Scale factors
         * @return Scaling matrix
         * @note Creates matrix that scales points by the specified factors
         */
        static float4x4 scaling(const float3& scale) noexcept;

        /**
         * @brief Uniform scaling matrix
         * @param uniformScale Uniform scale factor
         * @return Scaling matrix
         */
        static float4x4 scaling(float uniformScale) noexcept;

        /**
         * @brief Rotation matrix around X axis
         * @param angle Rotation angle in radians
         * @return Rotation matrix around X
         */
        static float4x4 rotation_x(float angle) noexcept;

        /**
         * @brief Rotation matrix around Y axis
         * @param angle Rotation angle in radians
         * @return Rotation matrix around Y
         */
        static float4x4 rotation_y(float angle) noexcept;

        /**
         * @brief Rotation matrix around Z axis
         * @param angle Rotation angle in radians
         * @return Rotation matrix around Z
         */
        static float4x4 rotation_z(float angle) noexcept;

        /**
         * @brief Rotation matrix from axis and angle
         * @param axis Rotation axis (must be normalized)
         * @param angle Rotation angle in radians
         * @return Rotation matrix
         * @note Uses Rodrigues' rotation formula
         */
        static float4x4 rotation_axis(const float3& axis, float angle) noexcept;

        /**
         * @brief Rotation matrix from Euler angles (ZYX order)
         * @param angles Euler angles in radians (pitch, yaw, roll)
         * @return Rotation matrix
         * @note Order: rotation_z(angles.z) * rotation_y(angles.y) * rotation_x(angles.x)
         */
        static float4x4 rotation_euler(const float3& angles) noexcept;

        /**
         * @brief TRS matrix (Scale * Rotation * Translation) for column-major
         * @param translation Translation vector
         * @param rotation Rotation quaternion
         * @param scale Scale vector
         * @return TRS transformation matrix
         * @note For column-major: M = T * R * S (applied in reverse order)
         */
        static float4x4 TRS(const float3& translation, const quaternion& rotation, const float3& scale) noexcept;

        // --- Projections (All Variants) ---

        /**
         * @brief Left-Handed Perspective projection with Zero-to-One depth range
         * @param fovY Vertical field of view in radians
         * @param aspect Aspect ratio (width/height)
         * @param zNear Near clipping plane
         * @param zFar Far clipping plane
         * @return Perspective projection matrix (LH, ZO)
         */
        static float4x4 perspective_lh_zo(float fovY, float aspect, float zNear, float zFar) noexcept;

        /**
         * @brief Right-Handed Perspective projection with Zero-to-One depth range
         * @param fovY Vertical field of view in radians
         * @param aspect Aspect ratio (width/height)
         * @param zNear Near clipping plane
         * @param zFar Far clipping plane
         * @return Perspective projection matrix (RH, ZO)
         * @note Vulkan / Modern OpenGL projection
         */
        static float4x4 perspective_rh_zo(float fovY, float aspect, float zNear, float zFar) noexcept;

        /**
         * @brief Left-Handed Perspective projection with Negative-One-to-One depth range
         * @param fovY Vertical field of view in radians
         * @param aspect Aspect ratio (width/height)
         * @param zNear Near clipping plane
         * @param zFar Far clipping plane
         * @return Perspective projection matrix (LH, NO)
         */
        static float4x4 perspective_lh_no(float fovY, float aspect, float zNear, float zFar) noexcept;

        /**
         * @brief Right-Handed Perspective projection with Negative-One-to-One depth range
         * @param fovY Vertical field of view in radians
         * @param aspect Aspect ratio (width/height)
         * @param zNear Near clipping plane
         * @param zFar Far clipping plane
         * @return Perspective projection matrix (RH, NO)
         * @note Legacy OpenGL projection
         */
        static float4x4 perspective_rh_no(float fovY, float aspect, float zNear, float zFar) noexcept;

        /**
         * @brief Default Perspective projection (RH, ZO)
         * @param fovY Vertical field of view in radians
         * @param aspect Aspect ratio (width/height)
         * @param zNear Near clipping plane
         * @param zFar Far clipping plane
         * @return Perspective projection matrix
         * @note Default to OpenGL/Vulkan style (RH, ZO)
         */
        static float4x4 perspective(float fovY, float aspect, float zNear, float zFar) noexcept;

        /**
         * @brief Left-Handed Orthographic projection with Zero-to-One depth range
         * @param width Viewport width
         * @param height Viewport height
         * @param zNear Near clipping plane
         * @param zFar Far clipping plane
         * @return Orthographic projection matrix (LH, ZO)
         */
        static float4x4 orthographic_lh_zo(float width, float height, float zNear, float zFar) noexcept;

        /**
         * @brief Left-Handed Off-Center Orthographic projection with Zero-to-One depth range
         * @param left Left clipping plane
         * @param right Right clipping plane
         * @param bottom Bottom clipping plane
         * @param top Top clipping plane
         * @param zNear Near clipping plane
         * @param zFar Far clipping plane
         * @return Orthographic projection matrix (LH, ZO)
         */
        static float4x4 orthographic_off_center_lh_zo(float left, float right, float bottom, float top, float zNear, float zFar) noexcept;

        /**
         * @brief Default Orthographic projection (LH, ZO)
         * @param width Viewport width
         * @param height Viewport height
         * @param zNear Near clipping plane
         * @param zFar Far clipping plane
         * @return Orthographic projection matrix
         */
        static float4x4 orthographic(float width, float height, float zNear, float zFar) noexcept;

        // --- Camera ---

        /**
         * @brief Left-Handed Look-At view matrix
         * @param eye Camera position
         * @param target Target position
         * @param up Up vector
         * @return View matrix (LH)
         */
        static float4x4 look_at_lh(const float3& eye, const float3& target, const float3& up) noexcept;

        /**
         * @brief Right-Handed Look-At view matrix
         * @param eye Camera position
         * @param target Target position
         * @param up Up vector
         * @return View matrix (RH)
         * @note OpenGL/Vulkan default view matrix
         */
        static float4x4 look_at_rh(const float3& eye, const float3& target, const float3& up) noexcept;

        /**
         * @brief Default Look-At view matrix (RH)
         * @param eye Camera position
         * @param target Target position
         * @param up Up vector
         * @return View matrix
         * @note Default to OpenGL/Vulkan style (RH)
         */
        static float4x4 look_at(const float3& eye, const float3& target, const float3& up) noexcept;

        // ============================================================================
        // Access Operators
        // ============================================================================

        /**
         * @brief Access column by index
         * @param colIndex Column index (0, 1, 2, or 3)
         * @return Reference to column
         * @note Column-major storage: [column][row]
         */
        float4& operator[](int colIndex) noexcept;

        /**
         * @brief Access column by index (const)
         * @param colIndex Column index (0, 1, 2, or 3)
         * @return Const reference to column
         */
        const float4& operator[](int colIndex) const noexcept;

        /**
         * @brief Access element by row and column (column-major)
         * @param row Row index (0, 1, 2, or 3)
         * @param col Column index (0, 1, 2, or 3)
         * @return Reference to element
         * @note Column-major: element at [row][column] = column[col][row]
         */
        float& operator()(int row, int col) noexcept;

        /**
         * @brief Access element by row and column (const)
         * @param row Row index (0, 1, 2, or 3)
         * @param col Column index (0, 1, 2, or 3)
         * @return Const reference to element
         */
        const float& operator()(int row, int col) const noexcept;

        // ============================================================================
        // Column and Row Accessors
        // ============================================================================

        /**
         * @brief Get column 0
         * @return First column
         */
        float4 col0() const noexcept { return col0_; }

        /**
         * @brief Get column 1
         * @return Second column
         */
        float4 col1() const noexcept { return col1_; }

        /**
         * @brief Get column 2
         * @return Third column
         */
        float4 col2() const noexcept { return col2_; }

        /**
         * @brief Get column 3
         * @return Fourth column (translation/perspective)
         */
        float4 col3() const noexcept { return col3_; }

        /**
         * @brief Set column 0
         * @param c New column values
         */
        void set_col0(const float4& c) { col0_ = c; }

        /**
         * @brief Set column 1
         * @param c New column values
         */
        void set_col1(const float4& c) { col1_ = c; }

        /**
         * @brief Set column 2
         * @param c New column values
         */
        void set_col2(const float4& c) { col2_ = c; }

        /**
         * @brief Set column 3
         * @param c New column values
         */
        void set_col3(const float4& c) { col3_ = c; }

        /**
         * @brief Get row 0
         * @return First row
         * @note Expensive in column-major storage
         */
        float4 row0() const noexcept;

        /**
         * @brief Get row 1
         * @return Second row
         */
        float4 row1() const noexcept;

        /**
         * @brief Get row 2
         * @return Third row
         */
        float4 row2() const noexcept;

        /**
         * @brief Get row 3
         * @return Fourth row
         */
        float4 row3() const noexcept;

        // ============================================================================
        // Compound Assignment Operators (SSE Optimized)
        // ============================================================================

        /**
         * @brief Matrix addition assignment
         * @param rhs Right-hand side matrix
         * @return Reference to this matrix
         * @note SSE optimized
         */
        float4x4& operator+=(const float4x4& rhs) noexcept;

        /**
         * @brief Matrix subtraction assignment
         * @param rhs Right-hand side matrix
         * @return Reference to this matrix
         * @note SSE optimized
         */
        float4x4& operator-=(const float4x4& rhs) noexcept;

        /**
         * @brief Scalar multiplication assignment
         * @param scalar Scalar multiplier
         * @return Reference to this matrix
         * @note SSE optimized
         */
        float4x4& operator*=(float scalar) noexcept;

        /**
         * @brief Scalar division assignment
         * @param scalar Scalar divisor
         * @return Reference to this matrix
         * @note SSE optimized
         */
        float4x4& operator/=(float scalar) noexcept;

        /**
         * @brief Matrix multiplication assignment
         * @param rhs Right-hand side matrix
         * @return Reference to this matrix
         * @note Uses full SSE optimization
         */
        float4x4& operator*=(const float4x4& rhs) noexcept;

        // ============================================================================
        // Unary Operators
        // ============================================================================

        /**
         * @brief Unary plus operator
         * @return Copy of this matrix
         */
        float4x4 operator+() const noexcept;

        /**
         * @brief Unary minus operator
         * @return Negated matrix
         */
        float4x4 operator-() const noexcept;

        // ============================================================================
        // Matrix Operations (SSE Optimized)
        // ============================================================================

        /**
         * @brief Compute transposed matrix
         * @return Transposed matrix
         * @note SSE optimized 4x4 transpose using shuffle operations
         */
        float4x4 transposed() const noexcept;

        /**
         * @brief Compute matrix determinant
         * @return Determinant value
         * @note Uses Laplace expansion for 4x4 determinant
         */
        float determinant() const noexcept;

        /**
         * @brief Compute inverse matrix for affine transformations
         * @return Inverse matrix
         * @note Optimized for affine matrices (last column = [0,0,0,1])
         */
        float4x4 inverted_affine() const noexcept;

        /**
         * @brief Compute inverse matrix
         * @return Inverse matrix
         * @note Returns identity if matrix is singular
         * @warning Matrix must be invertible (determinant != 0)
         */
        float4x4 inverted() const noexcept;

        /**
         * @brief Compute adjugate matrix
         * @return Adjugate matrix
         * @note Transpose of cofactor matrix, used for inverse calculation
         */
        float4x4 adjugate() const noexcept;

        /**
         * @brief Compute normal matrix (transpose(inverse(mat3x3)))
         * @return Normal transformation matrix
         * @note Used for transforming normal vectors in 3D graphics
         */
        float3x3 normal_matrix() const noexcept;

        /**
         * @brief Compute matrix trace (sum of diagonal elements)
         * @return Trace value
         */
        float trace() const noexcept;

        /**
         * @brief Extract diagonal elements
         * @return Diagonal as float4 vector
         */
        float4 diagonal() const noexcept;

        /**
         * @brief Compute Frobenius norm (sqrt of sum of squares of all elements)
         * @return Frobenius norm
         */
        float frobenius_norm() const noexcept;

        // ============================================================================
        // Vector Transformations (SSE Optimized)
        // ============================================================================

        /**
         * @brief Transform 4D vector (matrix * column vector)
         * @param vec 4D vector to transform
         * @return Transformed 4D vector
         * @note SSE optimized matrix-vector multiplication for column-major
         * @note Handles homogeneous coordinates and perspective division
         */
        float4 transform_vector(const float4& vec) const noexcept;

        /**
         * @brief Transform 3D point (applies translation and perspective)
         * @param point 3D point to transform
         * @return Transformed 3D point
         * @note Applies full homogeneous transformation with perspective division
         */
        float3 transform_point(const float3& point) const noexcept;

        /**
         * @brief Transform 3D vector (ignores translation)
         * @param vec 3D vector to transform
         * @return Transformed 3D vector
         * @note For vectors (w=0), ignores translation component
         */
        float3 transform_vector(const float3& vec) const noexcept;

        /**
         * @brief Transform 3D direction (normalizes result)
         * @param dir 3D direction to transform
         * @return Transformed and normalized 3D direction
         * @note Transforms direction vector and normalizes the result
         */
        float3 transform_direction(const float3& dir) const noexcept;

        // ============================================================================
        // Transformation Component Extraction
        // ============================================================================

        /**
         * @brief Extract translation component
         * @return Translation vector
         * @note Returns the XYZ components of the fourth column
         */
        float3 get_translation() const noexcept;

        /**
         * @brief Extract scale component
         * @return Scale vector
         * @note Computes length of each axis vector (columns 0-2)
         */
        float3 get_scale() const noexcept;

        /**
         * @brief Extract rotation component as quaternion
         * @return Rotation quaternion
         * @note Removes scaling and extracts pure rotation
         */
        quaternion get_rotation() const noexcept;

        /**
         * @brief Set translation component
         * @param translation New translation vector
         */
        void set_translation(const float3& translation) noexcept;

        /**
         * @brief Set scale component
         * @param scale New scale vector
         * @note Preserves existing rotation and translation
         */
        void set_scale(const float3& scale) noexcept;

        // ============================================================================
        // Utility Methods
        // ============================================================================

        /**
         * @brief Check if matrix is identity within tolerance
         * @param epsilon Comparison tolerance
         * @return True if matrix is approximately identity
         */
        bool is_identity(float epsilon = Constants::Constants<float>::Epsilon) const noexcept;

        /**
         * @brief Check if matrix is affine (last column is [0,0,0,1])
         * @param epsilon Comparison tolerance
         * @return True if matrix is affine
         * @note Affine matrices preserve parallel lines (no perspective)
         */
        bool is_affine(float epsilon = Constants::Constants<float>::Epsilon) const noexcept;

        /**
         * @brief Check if matrix is orthogonal
         * @param epsilon Comparison tolerance
         * @return True if matrix is orthogonal
         * @note Orthogonal matrices have orthonormal basis vectors
         */
        bool is_orthogonal(float epsilon = Constants::Constants<float>::Epsilon) const noexcept;

        /**
         * @brief Check approximate equality with another matrix
         * @param other Matrix to compare with
         * @param epsilon Comparison tolerance
         * @return True if matrices are approximately equal
         */
        bool approximately(const float4x4& other, float epsilon = Constants::Constants<float>::Epsilon) const noexcept;

        /**
         * @brief Check if matrix is approximately zero
         * @param epsilon Comparison tolerance
         * @return True if all matrix elements are approximately zero
         */
        bool approximately_zero(float epsilon = Constants::Constants<float>::Epsilon) const noexcept;

        /**
         * @brief Convert to string representation
         * @return String representation of matrix
         * @note Format: "[col0] [col1] [col2] [col3]" (column-major)
         */
        std::string to_string() const;

        /**
         * @brief Store matrix to column-major array
         * @param data Destination array (must have at least 16 elements)
         * @note Direct OpenGL/Vulkan compatible output
         */
        void to_column_major(float* data) const noexcept;

        /**
         * @brief Store matrix to row-major array
         * @param data Destination array (must have at least 16 elements)
         * @note Converts from column-major to row-major for DirectX
         */
        void to_row_major(float* data) const noexcept;

        // ============================================================================
        // Comparison Operators
        // ============================================================================

        /**
         * @brief Equality comparison
         * @param rhs Right-hand side matrix
         * @return True if matrices are approximately equal
         */
        bool operator==(const float4x4& rhs) const noexcept;

        /**
         * @brief Inequality comparison
         * @param rhs Right-hand side matrix
         * @return True if matrices are not approximately equal
         */
        bool operator!=(const float4x4& rhs) const noexcept;
    };

    // ============================================================================
    // Binary Operators
    // ============================================================================

    /**
     * @brief Matrix addition
     * @param lhs Left-hand side matrix
     * @param rhs Right-hand side matrix
     * @return Result of addition
     */
    inline float4x4 operator+(const float4x4& lhs, const float4x4& rhs) noexcept { return float4x4(lhs) += rhs; }

    /**
     * @brief Matrix subtraction
     * @param lhs Left-hand side matrix
     * @param rhs Right-hand side matrix
     * @return Result of subtraction
     */
    inline float4x4 operator-(const float4x4& lhs, const float4x4& rhs) noexcept { return float4x4(lhs) -= rhs; }

    /**
     * @brief Matrix multiplication
     * @param lhs Left-hand side matrix
     * @param rhs Right-hand side matrix
     * @return Product matrix
     * @note Full SSE optimized 4x4 matrix multiplication for column-major
     */
    float4x4 operator*(const float4x4& lhs, const float4x4& rhs) noexcept;

    /**
     * @brief Scalar multiplication
     * @param mat Matrix to scale
     * @param scalar Scalar multiplier
     * @return Scaled matrix
     */
    inline float4x4 operator*(const float4x4& mat, float scalar) noexcept { return float4x4(mat) *= scalar; }

    /**
     * @brief Scalar multiplication (commutative)
     * @param scalar Scalar multiplier
     * @param mat Matrix to scale
     * @return Scaled matrix
     */
    inline float4x4 operator*(float scalar, const float4x4& mat) noexcept { return float4x4(mat) *= scalar; }

    /**
     * @brief Scalar division
     * @param mat Matrix to divide
     * @param scalar Scalar divisor
     * @return Scaled matrix
     */
    inline float4x4 operator/(const float4x4& mat, float scalar) noexcept { return float4x4(mat) /= scalar; }

    /**
     * @brief Matrix-vector multiplication (matrix * column vector)
     * @param mat Transformation matrix
     * @param vec Vector to transform
     * @return Transformed vector
     * @note Column-major order: matrix * vector
     */
    inline float4 operator*(const float4x4& mat, const float4& vec) noexcept
    {
        return mat.transform_vector(vec);
    }

    /**
     * @brief Matrix-point multiplication (matrix * point)
     * @param mat Transformation matrix
     * @param point Point to transform
     * @return Transformed point
     * @note Applies homogeneous transformation with perspective division
     */
    inline float3 operator*(const float4x4& mat, const float3& point) noexcept
    {
        return mat.transform_point(point);
    }

    // ============================================================================
    // Global Functions
    // ============================================================================

    /**
     * @brief Compute transposed matrix
     * @param mat Matrix to transpose
     * @return Transposed matrix
     */
    inline float4x4 transpose(const float4x4& mat) noexcept { return mat.transposed(); }

    /**
     * @brief Compute inverse matrix
     * @param mat Matrix to invert
     * @return Inverse matrix
     */
    inline float4x4 inverse(const float4x4& mat) noexcept { return mat.inverted(); }

    /**
     * @brief Compute matrix determinant
     * @param mat Matrix to compute determinant of
     * @return Determinant value
     */
    inline float determinant(const float4x4& mat) noexcept { return mat.determinant(); }

    /**
     * @brief Matrix-vector multiplication
     * @param mat Transformation matrix
     * @param vec Vector to transform
     * @return Transformed vector
     */
    inline float4 mul(const float4x4& mat, const float4& vec) noexcept { return mat * vec; }

    /**
     * @brief Matrix-point multiplication
     * @param mat Transformation matrix
     * @param point Point to transform
     * @return Transformed point
     */
    inline float3 mul(const float4x4& mat, const float3& point) noexcept { return mat * point; }

    /**
     * @brief Matrix multiplication
     * @param lhs Left-hand side matrix
     * @param rhs Right-hand side matrix
     * @return Product matrix
     */
    inline float4x4 mul(const float4x4& lhs, const float4x4& rhs) noexcept { return lhs * rhs; }

    // ============================================================================
    // Useful Constants
    // ============================================================================

    /**
     * @brief Identity matrix constant
     */
    extern const float4x4 float4x4_Identity;

    /**
     * @brief Zero matrix constant
     */
    extern const float4x4 float4x4_Zero;

} // namespace Math

#include "math_float4x4.inl"
