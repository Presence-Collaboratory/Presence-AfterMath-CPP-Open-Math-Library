// Author: DeepSeek, NSDeathman
// Test suite for AfterMath::AABB class

#include "AutotestCore.h"

namespace AfterMathTests
{
    void RunAABBTests(TestSuite& suite)
    {
        using namespace AfterMath;

        // ============================================================================
        // 1. Constructors
        // ============================================================================
        suite.section("Constructors");

        // Default constructor (empty box: min=+inf, max=-inf)
        {
            AABB box;
            suite.assert_true(std::isinf(box.min.x) && box.min.x > 0.0f, "Default min.x is +inf");
            suite.assert_true(std::isinf(box.min.y) && box.min.y > 0.0f, "Default min.y is +inf");
            suite.assert_true(std::isinf(box.min.z) && box.min.z > 0.0f, "Default min.z is +inf");
            suite.assert_true(std::isinf(box.max.x) && box.max.x < 0.0f, "Default max.x is -inf");
            suite.assert_true(std::isinf(box.max.y) && box.max.y < 0.0f, "Default max.y is -inf");
            suite.assert_true(std::isinf(box.max.z) && box.max.z < 0.0f, "Default max.z is -inf");
            suite.assert_false(box.is_valid(), "Default AABB is invalid");
        }

        // Constructor with min/max
        {
            float3 mn(-1.0f, -2.0f, -3.0f);
            float3 mx(1.0f, 2.0f, 3.0f);
            AABB box(mn, mx);
            suite.assert_approximately_equal(box.min, mn, "Constructor min");
            suite.assert_approximately_equal(box.max, mx, "Constructor max");
            suite.assert_true(box.is_valid(), "Constructor with min<=max is valid");
        }

        // Copy constructor
        {
            AABB original(float3(-1.0f, -2.0f, -3.0f), float3(4.0f, 5.0f, 6.0f));
            AABB copy(original);
            suite.assert_approximately_equal(copy.min, original.min, "Copy min");
            suite.assert_approximately_equal(copy.max, original.max, "Copy max");
        }

        // ============================================================================
        // 2. Assignment operator
        // ============================================================================
        suite.section("Assignment operator");

        {
            AABB original(float3(-1.0f, -2.0f, -3.0f), float3(4.0f, 5.0f, 6.0f));
            AABB assigned;
            assigned = original;
            suite.assert_approximately_equal(assigned.min, original.min, "Assigned min");
            suite.assert_approximately_equal(assigned.max, original.max, "Assigned max");
        }

        // ============================================================================
        // 3. Static Constructors
        // ============================================================================
        suite.section("Static Constructors");

        // from_center_extents
        {
            float3 center(1.0f, 2.0f, 3.0f);
            float3 extents(0.5f, 1.0f, 1.5f);
            AABB box = AABB::from_center_extents(center, extents);
            suite.assert_approximately_equal(box.min, center - extents, "from_center_extents min");
            suite.assert_approximately_equal(box.max, center + extents, "from_center_extents max");
        }

        // from_points
        {
            float3 points[] = {
                float3(1.0f, 2.0f, 3.0f),
                float3(-1.0f, 5.0f, 0.0f),
                float3(0.5f, -2.0f, 4.0f),
                float3(2.0f, 1.0f, -1.0f)
            };
            AABB box = AABB::from_points(points, 4);
            suite.assert_approximately_equal(box.min, float3(-1.0f, -2.0f, -1.0f), "from_points min");
            suite.assert_approximately_equal(box.max, float3(2.0f, 5.0f, 4.0f), "from_points max");
        }

        // from_points with zero count
        {
            AABB box = AABB::from_points(nullptr, 0);
            suite.assert_false(box.is_valid(), "from_points with zero count returns invalid box");
        }

        // ============================================================================
        // 4. Basic Properties
        // ============================================================================
        suite.section("Basic Properties");

        {
            float3 mn(0.0f, 0.0f, 0.0f);
            float3 mx(4.0f, 6.0f, 8.0f);
            AABB box(mn, mx);

            suite.assert_approximately_equal(box.center(), float3(2.0f, 3.0f, 4.0f), "center()");
            suite.assert_approximately_equal(box.extents(), float3(2.0f, 3.0f, 4.0f), "extents()");
            suite.assert_approximately_equal(box.size(), float3(4.0f, 6.0f, 8.0f), "size()");
            suite.assert_true(box.is_valid(), "Valid box is valid");
        }

        // Invalid box (min > max)
        {
            AABB box(float3(1.0f, 1.0f, 1.0f), float3(0.0f, 0.0f, 0.0f));
            suite.assert_false(box.is_valid(), "min > max is invalid");
        }

        // Zero-size box (min == max) is valid
        {
            AABB box(float3(2.0f, 2.0f, 2.0f), float3(2.0f, 2.0f, 2.0f));
            suite.assert_true(box.is_valid(), "Zero-size box is valid");
        }

        // ============================================================================
        // 5. Expansion
        // ============================================================================
        suite.section("Expansion");

        // expand with point on empty box
        {
            AABB box;
            float3 p(1.0f, -2.0f, 3.0f);
            box.expand(p);
            suite.assert_approximately_equal(box.min, p, "After expand point, min = point");
            suite.assert_approximately_equal(box.max, p, "After expand point, max = point");
            suite.assert_true(box.is_valid(), "After expand point, box is valid");
        }

        // expand with multiple points
        {
            AABB box;
            box.expand(float3(1.0f, 2.0f, 3.0f));
            box.expand(float3(-1.0f, 5.0f, 0.0f));
            box.expand(float3(0.5f, -2.0f, 4.0f));
            box.expand(float3(2.0f, 1.0f, -1.0f));

            suite.assert_approximately_equal(box.min, float3(-1.0f, -2.0f, -1.0f), "expand points min");
            suite.assert_approximately_equal(box.max, float3(2.0f, 5.0f, 4.0f), "expand points max");
        }

        // expand with another AABB
        {
            AABB box1(float3(0.0f, 0.0f, 0.0f), float3(2.0f, 2.0f, 2.0f));
            AABB box2(float3(1.0f, -1.0f, 3.0f), float3(4.0f, 5.0f, 6.0f));
            box1.expand(box2);

            suite.assert_approximately_equal(box1.min, float3(0.0f, -1.0f, 0.0f), "expand AABB min");
            suite.assert_approximately_equal(box1.max, float3(4.0f, 5.0f, 6.0f), "expand AABB max");
        }

        // ============================================================================
        // 6. Containment Tests
        // ============================================================================
        suite.section("Containment Tests");

        // contains(point)
        {
            AABB box(float3(0.0f, 0.0f, 0.0f), float3(10.0f, 10.0f, 10.0f));

            suite.assert_true(box.contains(float3(5.0f, 5.0f, 5.0f)), "contains interior point");
            suite.assert_true(box.contains(float3(0.0f, 0.0f, 0.0f)), "contains min corner (inclusive)");
            suite.assert_true(box.contains(float3(10.0f, 10.0f, 10.0f)), "contains max corner (inclusive)");
            suite.assert_false(box.contains(float3(-1.0f, 5.0f, 5.0f)), "does not contain outside x");
            suite.assert_false(box.contains(float3(5.0f, 11.0f, 5.0f)), "does not contain outside y");
            suite.assert_false(box.contains(float3(5.0f, 5.0f, 10.1f)), "does not contain outside z");
        }

        // contains(AABB)
        {
            AABB outer(float3(0.0f, 0.0f, 0.0f), float3(10.0f, 10.0f, 10.0f));
            AABB inner(float3(2.0f, 2.0f, 2.0f), float3(8.0f, 8.0f, 8.0f));
            AABB same(float3(0.0f, 0.0f, 0.0f), float3(10.0f, 10.0f, 10.0f));
            AABB partially_out(float3(5.0f, 5.0f, 5.0f), float3(15.0f, 15.0f, 15.0f));
            AABB touch_edge(float3(0.0f, 0.0f, 0.0f), float3(10.0f, 10.0f, 10.0f));

            suite.assert_true(outer.contains(inner), "contains fully inside");
            suite.assert_true(outer.contains(same), "contains identical box");
            suite.assert_false(outer.contains(partially_out), "does not contain partially outside");
            suite.assert_true(outer.contains(touch_edge), "contains box sharing all boundaries (inclusive)");
        }

        // ============================================================================
        // 7. Intersects Tests
        // ============================================================================
        suite.section("Intersects Tests");

        {
            AABB a(float3(0.0f, 0.0f, 0.0f), float3(10.0f, 10.0f, 10.0f));
            AABB b(float3(5.0f, 5.0f, 5.0f), float3(15.0f, 15.0f, 15.0f));
            AABB c(float3(10.0f, 10.0f, 10.0f), float3(20.0f, 20.0f, 20.0f));
            AABB d(float3(11.0f, 11.0f, 11.0f), float3(20.0f, 20.0f, 20.0f));

            suite.assert_true(a.intersects(b), "Overlapping boxes intersect");
            suite.assert_true(a.intersects(c), "Touching at corner/edge intersects (inclusive)");
            suite.assert_false(a.intersects(d), "Separated boxes do not intersect");
        }

        // ============================================================================
        // 8. Ray Intersection
        // ============================================================================
        suite.section("Ray Intersection");

        {
            AABB box(float3(-1.0f, -1.0f, -1.0f), float3(1.0f, 1.0f, 1.0f));

            // Ray from outside hitting the box
            ray r1(float3(0.0f, 0.0f, -5.0f), float3(0.0f, 0.0f, 1.0f));
            float t1 = box.intersect(r1);
            suite.assert_approximately_equal(t1, 4.0f, "intersect ray from -z hits at t=4");

            // Alias intersect_ray
            float t_alias = box.intersect_ray(r1);
            suite.assert_approximately_equal(t_alias, 4.0f, "intersect_ray alias gives same result");

            // Ray starting inside the box
            ray r2(float3(0.0f, 0.0f, 0.0f), float3(0.0f, 0.0f, 1.0f));
            float t2 = box.intersect(r2);
            suite.assert_approximately_equal(t2, 0.0f, "ray inside box returns t=0");

            // Ray missing the box
            ray r3(float3(0.0f, 0.0f, -5.0f), float3(1.0f, 0.0f, 0.0f));
            float t3 = box.intersect(r3);
            suite.assert_approximately_equal(t3, -1.0f, "ray miss returns -1");

            // Ray parallel to slab but outside
            ray r4(float3(0.0f, 0.0f, 5.0f), float3(1.0f, 0.0f, 0.0f));
            float t4 = box.intersect(r4);
            suite.assert_approximately_equal(t4, -1.0f, "parallel ray outside slab returns -1");
        }

        // Global aabb_intersect_ray
        {
            AABB box(float3(-1.0f, -1.0f, -1.0f), float3(1.0f, 1.0f, 1.0f));
            ray r(float3(0.0f, 0.0f, -5.0f), float3(0.0f, 0.0f, 1.0f));
            suite.assert_approximately_equal(aabb_intersect_ray(box, r), 4.0f, "global aabb_intersect_ray works");
        }

        // ============================================================================
        // 9. Global Functions
        // ============================================================================
        suite.section("Global Functions");

        // aabb_from_center_extents
        {
            float3 center(1.0f, 2.0f, 3.0f);
            float3 extents(0.5f, 1.0f, 1.5f);
            AABB box = aabb_from_center_extents(center, extents);
            suite.assert_approximately_equal(box.min, center - extents, "aabb_from_center_extents min");
            suite.assert_approximately_equal(box.max, center + extents, "aabb_from_center_extents max");
        }

        // aabb_from_points
        {
            float3 pts[] = {
                float3(1.0f, 2.0f, 3.0f),
                float3(-1.0f, 5.0f, 0.0f),
                float3(0.5f, -2.0f, 4.0f)
            };
            AABB box = aabb_from_points(pts, 3);
            suite.assert_approximately_equal(box.min, float3(-1.0f, -2.0f, 0.0f), "aabb_from_points min");
            suite.assert_approximately_equal(box.max, float3(1.0f, 5.0f, 4.0f), "aabb_from_points max");
        }

        // aabb_union
        {
            AABB a(float3(0.0f, 0.0f, 0.0f), float3(2.0f, 2.0f, 2.0f));
            AABB b(float3(1.0f, -1.0f, 3.0f), float3(4.0f, 5.0f, 6.0f));
            AABB u = aabb_union(a, b);
            suite.assert_approximately_equal(u.min, float3(0.0f, -1.0f, 0.0f), "aabb_union min");
            suite.assert_approximately_equal(u.max, float3(4.0f, 5.0f, 6.0f), "aabb_union max");
        }

        // aabb_intersects
        {
            AABB a(float3(0.0f, 0.0f, 0.0f), float3(10.0f, 10.0f, 10.0f));
            AABB b(float3(5.0f, 5.0f, 5.0f), float3(15.0f, 15.0f, 15.0f));
            AABB c(float3(20.0f, 20.0f, 20.0f), float3(30.0f, 30.0f, 30.0f));
            suite.assert_true(aabb_intersects(a, b), "aabb_intersects true for overlapping");
            suite.assert_false(aabb_intersects(a, c), "aabb_intersects false for separate");
        }

        // ============================================================================
        // 10. Comparison Operators
        // ============================================================================
        suite.section("Comparison Operators");

        {
            AABB a(float3(0.0f, 0.0f, 0.0f), float3(10.0f, 10.0f, 10.0f));
            AABB b(float3(0.0f, 0.0f, 0.0f), float3(10.0f, 10.0f, 10.0f));
            AABB c(float3(0.0f, 0.0f, 0.0f), float3(11.0f, 10.0f, 10.0f));

            suite.assert_true(a == b, "Equal AABBs");
            suite.assert_false(a != b, "Equal AABBs not unequal");
            suite.assert_false(a == c, "Different AABBs");
            suite.assert_true(a != c, "Different AABBs unequal");
        }

        // approximately
        {
            AABB a(float3(0.0f, 0.0f, 0.0f), float3(10.0f, 10.0f, 10.0f));
            AABB b(float3(0.000001f, 0.0f, 0.0f), float3(10.0f, 10.000001f, 10.0f));
            suite.assert_true(approximately(a, b, 1e-5f), "approximately within epsilon");
            suite.assert_false(approximately(a, b, 1e-8f), "approximately not within small epsilon");
        }

        // ============================================================================
        // 11. Edge Cases
        // ============================================================================
        suite.section("Edge Cases");

        // Empty box after default is invalid and can be expanded
        {
            AABB box;
            suite.assert_false(box.is_valid(), "Default box invalid");
            box.expand(float3(0.0f, 0.0f, 0.0f));
            suite.assert_true(box.is_valid(), "After expand point, valid");
            suite.assert_approximately_equal(box.min, box.max, "After single expand, min==max");
        }

        // Box with negative size (invalid)
        {
            AABB box(float3(10.0f, 10.0f, 10.0f), float3(0.0f, 0.0f, 0.0f));
            suite.assert_false(box.is_valid(), "Negative size invalid");
            suite.assert_false(box.contains(float3(5.0f, 5.0f, 5.0f)), "Invalid box contains nothing");
        }

        // Ray hitting AABB edge (t should be exact)
        {
            AABB box(float3(0.0f, 0.0f, 0.0f), float3(2.0f, 2.0f, 2.0f));
            ray r(float3(1.0f, 1.0f, -1.0f), float3(0.0f, 0.0f, 1.0f));
            float t = box.intersect(r);
            suite.assert_approximately_equal(t, 1.0f, "ray hits front face at t=1");
        }

        // AABB with zero extent (point) containment
        {
            AABB point_box(float3(3.0f, 4.0f, 5.0f), float3(3.0f, 4.0f, 5.0f));
            suite.assert_true(point_box.is_valid(), "Zero-extent box valid");
            suite.assert_true(point_box.contains(float3(3.0f, 4.0f, 5.0f)), "contains point itself");
            suite.assert_false(point_box.contains(float3(3.1f, 4.0f, 5.0f)), "does not contain nearby point");
        }

        // to_string smoke test
        {
            AABB box(float3(1.0f, 2.0f, 3.0f), float3(4.0f, 5.0f, 6.0f));
            std::string s = box.to_string();
            suite.assert_true(s.find("AABB") != std::string::npos, "to_string contains 'AABB'");
            suite.assert_true(s.find("1.0") != std::string::npos || s.find("1.000") != std::string::npos, "to_string contains min.x");
            suite.assert_true(s.find("6.0") != std::string::npos || s.find("6.000") != std::string::npos, "to_string contains max.z");
        }
    }
}
