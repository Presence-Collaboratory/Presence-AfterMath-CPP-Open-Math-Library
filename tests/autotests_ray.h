// Author: DeepSeek, NSDeathman
// Test suite for AfterMath::ray class

#include "AutotestCore.h"

namespace AfterMathTests
{
    void RunRayTests(TestSuite& suite)
    {
        using namespace AfterMath;

        // ============================================================================
        // 1. Constructors
        // ============================================================================
        suite.section("Constructors");

        // Default constructor
        {
            ray r;
            suite.assert_approximately_equal(r.origin, float3::zero(), "Default origin is zero");
            suite.assert_approximately_equal(r.direction, float3::forward(), "Default direction is forward");
        }

        // Parameterized constructor
        {
            float3 origin(1.0f, 2.0f, 3.0f);
            float3 direction(0.0f, 0.0f, 1.0f);
            ray r(origin, direction);
            suite.assert_approximately_equal(r.origin, origin, "Parameterized origin");
            suite.assert_approximately_equal(r.direction, direction, "Parameterized direction");
        }

        // Copy constructor
        {
            ray original(float3(1.0f, 2.0f, 3.0f), float3(0.0f, 1.0f, 0.0f));
            ray copy(original);
            suite.assert_approximately_equal(copy.origin, original.origin, "Copy origin");
            suite.assert_approximately_equal(copy.direction, original.direction, "Copy direction");
        }

        // ============================================================================
        // 2. Assignment operator
        // ============================================================================
        suite.section("Assignment operator");

        {
            ray original(float3(1.0f, 2.0f, 3.0f), float3(0.0f, 1.0f, 0.0f));
            ray assigned;
            assigned = original;
            suite.assert_approximately_equal(assigned.origin, original.origin, "Assigned origin");
            suite.assert_approximately_equal(assigned.direction, original.direction, "Assigned direction");
        }

        // ============================================================================
        // 3. Methods
        // ============================================================================
        suite.section("Methods");

        // point_at
        {
            float3 origin(1.0f, 2.0f, 3.0f);
            float3 direction(0.0f, 0.0f, 1.0f);
            ray r(origin, direction);

            suite.assert_approximately_equal(r.point_at(5.0f), float3(1.0f, 2.0f, 8.0f), "point_at(5)");
            suite.assert_approximately_equal(r.point_at(0.0f), origin, "point_at(0) returns origin");
            suite.assert_approximately_equal(r.point_at(-2.0f), float3(1.0f, 2.0f, 1.0f), "point_at(-2) behind origin");
        }

        // at (alias)
        {
            ray r(float3(0.0f, 0.0f, 0.0f), float3(1.0f, 0.0f, 0.0f));
            suite.assert_approximately_equal(r.at(3.0f), float3(3.0f, 0.0f, 0.0f), "at(3) alias for point_at");
        }

        // normalized
        {
            float3 origin(1.0f, 2.0f, 3.0f);
            float3 dir(0.0f, 3.0f, 4.0f); // length = 5
            ray r(origin, dir);
            ray normalized = r.normalized();
            suite.assert_approximately_equal(normalized.origin, origin, "normalized origin unchanged");
            suite.assert_approximately_equal(length(normalized.direction), 1.0f, "normalized direction length is 1");
            suite.assert_approximately_equal(normalized.direction, normalize(dir), "normalized direction matches normalize(dir)");
        }

        // normalized with zero direction
        {
            ray r(float3(1.0f, 2.0f, 3.0f), float3::zero());
            ray normalized = r.normalized();
            suite.assert_approximately_equal(normalized.direction, float3::zero(), "normalized zero direction returns zero");
        }

        // transformed with identity matrix
        {
            ray r(float3(1.0f, 2.0f, 3.0f), float3(0.0f, 0.0f, 1.0f));
            ray transformed = r.transformed(float4x4_Identity);
            suite.assert_approximately_equal(transformed.origin, r.origin, "transformed identity origin unchanged");
            suite.assert_approximately_equal(transformed.direction, r.direction, "transformed identity direction unchanged");
        }

        // transform in-place with identity
        {
            ray r(float3(1.0f, 2.0f, 3.0f), float3(0.0f, 0.0f, 1.0f));
            ray original = r;
            r.transform(float4x4_Identity);
            suite.assert_approximately_equal(r.origin, original.origin, "transform in-place origin unchanged");
            suite.assert_approximately_equal(r.direction, original.direction, "transform in-place direction unchanged");
        }

        // to_string smoke test
        {
            ray r(float3(1.0f, 2.0f, 3.0f), float3(0.0f, 0.0f, 1.0f));
            std::string s = r.to_string();
            suite.assert_true(s.find("Ray") != std::string::npos, "to_string contains 'Ray'");
            suite.assert_true(s.find("1.0") != std::string::npos || s.find("1.000") != std::string::npos, "to_string contains origin x");
        }

        // ============================================================================
        // 4. Global ray functions
        // ============================================================================
        suite.section("Global ray functions");

        // ray_from_points
        {
            float3 start(1.0f, 2.0f, 3.0f);
            float3 end(1.0f, 2.0f, 5.0f);
            ray r = ray_from_points(start, end);
            suite.assert_approximately_equal(r.origin, start, "ray_from_points origin");
            suite.assert_approximately_equal(r.direction, float3(0.0f, 0.0f, 1.0f), "ray_from_points normalized direction");
        }

        // ray_from_points with identical points
        {
            float3 p(1.0f, 2.0f, 3.0f);
            ray r = ray_from_points(p, p);
            suite.assert_approximately_equal(r.origin, p, "ray_from_points identical origin");
            suite.assert_approximately_equal(r.direction, float3::zero(), "ray_from_points identical direction zero");
        }

        // point_on_ray
        {
            float3 origin(0.0f, 0.0f, 0.0f);
            float3 direction(1.0f, 0.0f, 0.0f);
            ray r(origin, direction);
            suite.assert_approximately_equal(point_on_ray(r, 2.5f), float3(2.5f, 0.0f, 0.0f), "point_on_ray at 2.5");
        }

        // ============================================================================
        // 5. Intersection tests
        // ============================================================================
        suite.section("Intersection tests");

        // Plane intersection
        {
            // Ray hits plane z = 5 at t = 5
            ray r(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 1.0f));
            float4 plane(0.0f, 0.0f, 1.0f, -5.0f); // plane z = 5
            float t = intersect_ray_plane(r, plane);
            suite.assert_approximately_equal(t, 5.0f, "Plane hit forward");
        }

        {
            // Ray parallel to plane
            ray r(float3(0.0f, 0.0f, 1.0f), float3(1.0f, 0.0f, 0.0f));
            float4 plane(0.0f, 0.0f, 1.0f, -5.0f);
            float t = intersect_ray_plane(r, plane);
            suite.assert_approximately_equal(t, -1.0f, "Plane parallel no hit");
        }

        {
            // Ray facing away from plane
            ray r(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, -1.0f));
            float4 plane(0.0f, 0.0f, 1.0f, -5.0f); // plane z = 5
            float t = intersect_ray_plane(r, plane);
            suite.assert_approximately_equal(t, -1.0f, "Plane facing away");
        }

        {
            // Ray origin on plane (should return t=0)
            ray r(float3(0.0f, 0.0f, 5.0f), float3(0.0f, 0.0f, 1.0f));
            float4 plane(0.0f, 0.0f, 1.0f, -5.0f);
            float t = intersect_ray_plane(r, plane);
            suite.assert_approximately_equal(t, 0.0f, "Plane origin on plane returns 0");
        }

        // Sphere intersection
        {
            // Ray hits sphere center (0,0,5), radius 1, from origin (0,0,0) along +z
            ray r(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 1.0f));
            float3 center(0.0f, 0.0f, 5.0f);
            float t = intersect_ray_sphere(r, center, 1.0f);
            suite.assert_approximately_equal(t, 4.0f, "Sphere hit at t=4");
        }

        {
            // Ray starts inside sphere
            ray r(float3(0.0f, 0.0f, 5.0f), float3(0.0f, 0.0f, 1.0f));
            float3 center(0.0f, 0.0f, 5.0f);
            float t = intersect_ray_sphere(r, center, 2.0f);
            suite.assert_approximately_equal(t, 2.0f, "Sphere origin inside returns exit t");
        }

        {
            // Ray misses sphere
            ray r(float3(0.0f, 0.0f, 0.0f), float3(1.0f, 0.0f, 0.0f));
            float3 center(0.0f, 0.0f, 5.0f);
            float t = intersect_ray_sphere(r, center, 1.0f);
            suite.assert_approximately_equal(t, -1.0f, "Sphere miss returns -1");
        }

        {
            // Ray points away from sphere, but sphere is behind (direction towards it)
            ray r(float3(0.0f, 0.0f, 10.0f), float3(0.0f, 0.0f, -1.0f));
            float3 center(0.0f, 0.0f, 5.0f);
            float t = intersect_ray_sphere(r, center, 1.0f);
            suite.assert_approximately_equal(t, 4.0f, "Sphere behind hit when pointing towards");
        }

        {
            // Ray origin on sphere surface pointing outward
            ray r(float3(0.0f, 0.0f, 6.0f), float3(0.0f, 0.0f, 1.0f));
            float3 center(0.0f, 0.0f, 5.0f);
            float t = intersect_ray_sphere(r, center, 1.0f);
            suite.assert_approximately_equal(t, 0.0f, "Sphere origin on surface returns 0");
        }

        // AABB intersection
        {
            // Ray hits AABB
            ray r(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 1.0f));
            float3 min_bound(-1.0f, -1.0f, 5.0f);
            float3 max_bound(1.0f, 1.0f, 6.0f);
            float t = intersect_ray_aabb(r, min_bound, max_bound);
            suite.assert_approximately_equal(t, 5.0f, "AABB hit at t=5");
        }

        {
            // Ray origin inside AABB
            ray r(float3(0.0f, 0.0f, 5.5f), float3(0.0f, 0.0f, 1.0f));
            float3 min_bound(-1.0f, -1.0f, 5.0f);
            float3 max_bound(1.0f, 1.0f, 6.0f);
            float t = intersect_ray_aabb(r, min_bound, max_bound);
            suite.assert_approximately_equal(t, 0.0f, "AABB origin inside returns 0");
        }

        {
            // Ray misses AABB
            ray r(float3(0.0f, 0.0f, 0.0f), float3(1.0f, 0.0f, 0.0f));
            float3 min_bound(-1.0f, -1.0f, 5.0f);
            float3 max_bound(1.0f, 1.0f, 6.0f);
            float t = intersect_ray_aabb(r, min_bound, max_bound);
            suite.assert_approximately_equal(t, -1.0f, "AABB miss returns -1");
        }

        {
            // Parallel ray outside slab
            ray r(float3(0.0f, 0.0f, 0.0f), float3(1.0f, 0.0f, 0.0f));
            float3 min_bound(-1.0f, -1.0f, 5.0f);
            float3 max_bound(1.0f, 1.0f, 6.0f);
            // z = 0 is outside [5,6], so should miss
            float t = intersect_ray_aabb(r, min_bound, max_bound);
            suite.assert_approximately_equal(t, -1.0f, "AABB parallel outside slab");
        }

        // ============================================================================
        // 6. Comparison operators
        // ============================================================================
        suite.section("Comparison operators");

        {
            ray a(float3(1.0f, 2.0f, 3.0f), float3(0.0f, 0.0f, 1.0f));
            ray b(float3(1.0f, 2.0f, 3.0f), float3(0.0f, 0.0f, 1.0f));
            ray c(float3(1.0f, 2.0f, 3.0f), float3(0.0f, 1.0f, 0.0f));
            ray d(float3(1.1f, 2.0f, 3.0f), float3(0.0f, 0.0f, 1.0f));

            suite.assert_true(a == b, "Equal rays");
            suite.assert_false(a != b, "Equal rays not unequal");
            suite.assert_false(a == c, "Different direction");
            suite.assert_true(a != c, "Different direction unequal");
            suite.assert_false(a == d, "Different origin");
            suite.assert_true(a != d, "Different origin unequal");
        }

        {
            // approximately
            ray a(float3(1.0f, 2.0f, 3.0f), float3(0.0f, 0.0f, 1.0f));
            ray b(float3(1.000001f, 2.000001f, 3.000001f), float3(0.0f, 0.0f, 1.000001f));
            ray c(float3(1.0f, 2.0f, 3.0f), float3(0.0f, 0.0f, 1.0f));
            suite.assert_true(approximately(a, b, 1e-5f), "approximately within epsilon");
            suite.assert_true(approximately(a, c, 1e-8f), "approximately identical with small epsilon");
        }
    }
}
