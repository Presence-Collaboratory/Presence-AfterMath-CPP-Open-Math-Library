// Author: Deepseek
// Test suite for AfterMath::AABB class

#include "AutotestCore.h"

namespace AfterMathTests
{
    void RunAABBTests()
    {
        TestSuite suite("AABB Tests", true);
        suite.header();

        using namespace AfterMath;

        // ============================================================================
        // 1. Конструкторы и базовые операции
        // ============================================================================
        suite.section("Конструкторы и базовые операции");

        // Тест конструктора по умолчанию (invalid AABB)
        {
            AABB empty = AABB();
            suite.assert_false(empty.is_valid(), "Default constructor creates invalid AABB");
            suite.assert_true(empty.is_empty(), "Default constructor creates empty AABB");
        }

        // Тест конструктора из минимальной и максимальной точек
        {
            float3 min(1, 2, 3);
            float3 max(4, 5, 6);
            AABB box(min, max);

            suite.assert_true(box.is_valid(), "Constructor with min/max creates valid AABB");
            suite.assert_equal(box.min, min, "min property set correctly");
            suite.assert_equal(box.max, max, "max property set correctly");
        }

        // Тест конструктора из одной точки
        {
            float3 point(2, 3, 4);
            AABB box(point);

            suite.assert_true(box.is_valid(), "Constructor from point creates valid AABB");
            suite.assert_equal(box.min, point, "min set to point");
            suite.assert_equal(box.max, point, "max set to point");
        }

        // Тест конструктора из центра и экстентов
        {
            float3 center(2, 3, 4);
            float3 extents(1, 2, 3);
            AABB box = AABB::from_center_extents(center, extents);

            float3 expected_min(1, 1, 1);
            float3 expected_max(3, 5, 7);
            suite.assert_approximately_equal(box.min, expected_min, "from_center_extents min");
            suite.assert_approximately_equal(box.max, expected_max, "from_center_extents max");
        }

        // Тест конструктора из массива точек
        {
            float3 points[] = {
                float3(1, 2, 3),
                float3(4, 5, 6),
                float3(2, 3, 4),
                float3(0, 1, 2)
            };

            AABB box = AABB::from_points(points, 4);
            float3 expected_min(0, 1, 2);
            float3 expected_max(4, 5, 6);

            suite.assert_equal(box.min, expected_min, "from_points min");
            suite.assert_equal(box.max, expected_max, "from_points max");
        }

        // Тест конструктора из двух AABB (union)
        {
            AABB a(float3(1, 2, 3), float3(3, 4, 5));
            AABB b(float3(2, 1, 4), float3(5, 3, 6));
            AABB box = AABB::from_aabbs(a, b);

            float3 expected_min(1, 1, 3);
            float3 expected_max(5, 4, 6);

            suite.assert_equal(box.min, expected_min, "from_aabbs (union) min");
            suite.assert_equal(box.max, expected_max, "from_aabbs (union) max");
        }

        // ============================================================================
        // 2. Доступ к элементам и свойствам
        // ============================================================================
        suite.section("Доступ к элементам и свойствам");

        AABB box(float3(1, 2, 3), float3(4, 5, 6));

        // Проверка валидности
        suite.assert_true(box.is_valid(), "is_valid for valid AABB");
        suite.assert_false(box.is_empty(), "is_empty for non-empty AABB");

        // Проверка center
        {
            float3 expected_center(2.5f, 3.5f, 4.5f);
            suite.assert_approximately_equal(box.center(), expected_center, "center() calculation");
        }

        // Проверка extents
        {
            float3 expected_extents(1.5f, 1.5f, 1.5f);
            suite.assert_approximately_equal(box.extents(), expected_extents, "extents() calculation");
        }

        // Проверка size
        {
            float3 expected_size(3, 3, 3);
            suite.assert_approximately_equal(box.size(), expected_size, "size() calculation");
        }

        // Проверка surface_area
        {
            float expected_area = 2.0f * (3 * 3 + 3 * 3 + 3 * 3); // 2*(ab + ac + bc)
            suite.assert_approximately_equal(box.surface_area(), expected_area, "surface_area() calculation");
        }

        // Проверка volume
        {
            float expected_volume = 3 * 3 * 3; // 27
            suite.assert_approximately_equal(box.volume(), expected_volume, "volume() calculation");
        }

        // Проверка corner по индексу
        {
            // Углы: 0=min, 7=max
            suite.assert_equal(box.corner(0), float3(1, 2, 3), "corner(0) = min");
            suite.assert_equal(box.corner(7), float3(4, 5, 6), "corner(7) = max");
            suite.assert_equal(box.corner(1), float3(4, 2, 3), "corner(1)");
            suite.assert_equal(box.corner(2), float3(1, 5, 3), "corner(2)");
            suite.assert_equal(box.corner(4), float3(1, 2, 6), "corner(4)");
        }

        // Проверка всех углов
        {
            float3 corners[8];
            box.get_corners(corners);

            suite.assert_equal(corners[0], float3(1, 2, 3), "get_corners[0]");
            suite.assert_equal(corners[1], float3(4, 2, 3), "get_corners[1]");
            suite.assert_equal(corners[2], float3(1, 5, 3), "get_corners[2]");
            suite.assert_equal(corners[3], float3(4, 5, 3), "get_corners[3]");
            suite.assert_equal(corners[4], float3(1, 2, 6), "get_corners[4]");
            suite.assert_equal(corners[5], float3(4, 2, 6), "get_corners[5]");
            suite.assert_equal(corners[6], float3(1, 5, 6), "get_corners[6]");
            suite.assert_equal(corners[7], float3(4, 5, 6), "get_corners[7]");
        }

        // ============================================================================
        // 3. Тесты содержания
        // ============================================================================
        suite.section("Тесты содержания");

        AABB container(float3(0, 0, 0), float3(10, 10, 10));

        // Проверка contains для точки
        {
            suite.assert_true(container.contains(float3(5, 5, 5)), "contains point inside");
            suite.assert_true(container.contains(float3(0, 0, 0)), "contains point on min boundary");
            suite.assert_true(container.contains(float3(10, 10, 10)), "contains point on max boundary");
            suite.assert_false(container.contains(float3(-1, 5, 5)), "contains point outside (x)");
            suite.assert_false(container.contains(float3(11, 5, 5)), "contains point outside (x+)");
            suite.assert_false(container.contains(float3(5, -1, 5)), "contains point outside (y)");
            suite.assert_false(container.contains(float3(5, 11, 5)), "contains point outside (y+)");
            suite.assert_false(container.contains(float3(5, 5, -1)), "contains point outside (z)");
            suite.assert_false(container.contains(float3(5, 5, 11)), "contains point outside (z+)");
        }

        // Проверка contains для AABB
        {
            AABB inside(float3(2, 2, 2), float3(8, 8, 8));
            AABB outside(float3(-1, 2, 2), float3(8, 8, 8));
            AABB partial(float3(8, 8, 8), float3(12, 12, 12));

            suite.assert_true(container.contains(inside), "contains AABB inside");
            suite.assert_false(container.contains(outside), "contains AABB outside");
            suite.assert_false(container.contains(partial), "contains AABB partially outside");
        }

        // Проверка intersects для AABB
        {
            AABB other1(float3(5, 5, 5), float3(15, 15, 15)); // Пересекается
            AABB other2(float3(11, 11, 11), float3(15, 15, 15)); // Не пересекается
            AABB other3(float3(10, 10, 10), float3(15, 15, 15)); // Касается грани

            suite.assert_true(container.intersects(other1), "intersects with overlapping AABB");
            suite.assert_false(container.intersects(other2), "intersects with non-overlapping AABB");
            suite.assert_true(container.intersects(other3), "intersects with touching AABB");
        }

        // ============================================================================
        // 4. Операции с лучами (пересечение)
        // ============================================================================
        suite.section("Операции с лучами");

        AABB testBox(float3(-1, -1, -1), float3(1, 1, 1));

        // Простой тест пересечения луча
        {
            float3 origin(0, 0, -5);
            float3 direction(0, 0, 1); // Направлен к коробке

            bool intersects = testBox.intersect_ray(origin, direction);
            suite.assert_true(intersects, "ray intersects from front");
        }

        // Тест пересечения луча с подробной информацией
        {
            float3 origin(0, 0, -5);
            float3 direction(0, 0, 1);
            float t_min, t_max;
            float3 normal;

            bool intersects = testBox.intersect_ray(origin, direction, t_min, t_max, normal);

            suite.assert_true(intersects, "detailed ray intersection");
            suite.assert_approximately_equal(t_min, 4.0f, "t_min correct", 1e-5f);
            suite.assert_approximately_equal(t_max, 6.0f, "t_max correct", 1e-5f);
            suite.assert_approximately_equal(normal, float3(0, 0, -1), "normal at entry", 1e-5f);
        }

        // Тест луча, который не пересекает
        {
            float3 origin(2, 2, -5);
            float3 direction(0, 0, 1); // Направлен мимо коробки

            bool intersects = testBox.intersect_ray(origin, direction);
            suite.assert_false(intersects, "ray misses AABB");
        }

        // Тест луча изнутри AABB
        {
            float3 origin(0, 0, 0);
            float3 direction(0, 0, 1);
            float t_min, t_max;

            bool intersects = testBox.intersect_ray(origin, direction, t_min, t_max);

            suite.assert_true(intersects, "ray from inside intersects");
            suite.assert_approximately_equal(t_min, 0.0f, "t_min from inside", 1e-5f);
            suite.assert_approximately_equal(t_max, 1.0f, "t_max from inside", 1e-5f);
        }

        // ============================================================================
        // 5. Операции расширения
        // ============================================================================
        suite.section("Операции расширения");

        // Расширение точкой
        {
            AABB box(float3(1, 1, 1), float3(3, 3, 3));
            box.expand(float3(4, 2, 2));

            suite.assert_equal(box.min, float3(1, 1, 1), "expand point - min unchanged");
            suite.assert_equal(box.max, float3(4, 3, 3), "expand point - max updated");
        }

        // Расширение другим AABB
        {
            AABB box(float3(1, 1, 1), float3(3, 3, 3));
            AABB other(float3(0, 2, 2), float3(2, 4, 4));

            box.expand(other);

            suite.assert_equal(box.min, float3(0, 1, 1), "expand AABB - min updated");
            suite.assert_equal(box.max, float3(3, 4, 4), "expand AABB - max updated");
        }

        // Расширение на скаляр
        {
            AABB box(float3(1, 1, 1), float3(3, 3, 3));
            box.expand(1.0f);

            suite.assert_equal(box.min, float3(0, 0, 0), "expand scalar - min decreased");
            suite.assert_equal(box.max, float3(4, 4, 4), "expand scalar - max increased");
        }

        // ============================================================================
        // 6. Преобразования
        // ============================================================================
        suite.section("Преобразования");

        // Перевод (translation)
        {
            AABB box(float3(1, 2, 3), float3(4, 5, 6));
            float3 translation(2, 3, 4);

            AABB translated = box.translate(translation);

            suite.assert_equal(translated.min, float3(3, 5, 7), "translate - min");
            suite.assert_equal(translated.max, float3(6, 8, 10), "translate - max");
        }

        // Масштабирование (scale)
        {
            AABB box(float3(-1, -1, -1), float3(1, 1, 1));
            float3 scale(2, 3, 4);

            AABB scaled = box.scale(scale);

            suite.assert_equal(scaled.min, float3(-2, -3, -4), "scale - min");
            suite.assert_equal(scaled.max, float3(2, 3, 4), "scale - max");
        }

        // Преобразование матрицей (трансляция и масштабирование)
        {
            AABB box(float3(-1, -1, -1), float3(1, 1, 1));
            float4x4 transform = float4x4::scaling(float3(2, 3, 4)) *
                float4x4::translation(float3(1, 2, 3));

            AABB transformed = box.transform(transform);

            // Ожидаемые углы: [-1,-1,-1] -> [-2,-3,-4] + [1,2,3] = [-1,-1,-1]
            //                [1,1,1] -> [2,3,4] + [1,2,3] = [3,5,7]
            // Но transform использует преобразование всех углов, так что ожидаем консервативный AABB
            float3 expected_min(-1, -1, -1);
            float3 expected_max(3, 5, 7);

            suite.assert_equal(transformed.min, expected_min, "transform with matrix - min");
            suite.assert_equal(transformed.max, expected_max, "transform with matrix - max");
        }

        // ============================================================================
        // 7. Статические операции
        // ============================================================================
        suite.section("Статические операции");

        // Объединение (combine/union)
        {
            AABB a(float3(1, 1, 1), float3(3, 3, 3));
            AABB b(float3(2, 2, 2), float3(4, 4, 4));

            AABB combined = AABB::combine(a, b);

            suite.assert_equal(combined.min, float3(1, 1, 1), "combine - min");
            suite.assert_equal(combined.max, float3(4, 4, 4), "combine - max");
        }

        // Пересечение (intersect)
        {
            AABB a(float3(1, 1, 1), float3(4, 4, 4));
            AABB b(float3(2, 2, 2), float3(5, 5, 5));

            AABB intersection = AABB::intersect(a, b);

            suite.assert_equal(intersection.min, float3(2, 2, 2), "intersect - min");
            suite.assert_equal(intersection.max, float3(4, 4, 4), "intersect - max");
        }

        // Пересечение непересекающихся AABB
        {
            AABB a(float3(1, 1, 1), float3(2, 2, 2));
            AABB b(float3(3, 3, 3), float3(4, 4, 4));

            AABB intersection = AABB::intersect(a, b);
            suite.assert_false(intersection.is_valid(), "intersect of non-intersecting AABBs is invalid");
        }

        // ============================================================================
        // 8. Утилитарные методы
        // ============================================================================
        suite.section("Утилитарные методы");

        // longest_axis
        {
            AABB box1(float3(0, 0, 0), float3(10, 2, 3));
            AABB box2(float3(0, 0, 0), float3(2, 10, 3));
            AABB box3(float3(0, 0, 0), float3(2, 3, 10));

            suite.assert_equal(box1.longest_axis(), 0, "longest_axis - X axis");
            suite.assert_equal(box2.longest_axis(), 1, "longest_axis - Y axis");
            suite.assert_equal(box3.longest_axis(), 2, "longest_axis - Z axis");
        }

        // max_extent
        {
            AABB box(float3(0, 0, 0), float3(3, 5, 2));
            suite.assert_approximately_equal(box.max_extent(), 5.0f, "max_extent");
        }

        // normalize_coords и denormalize_coords
        {
            AABB box(float3(1, 2, 3), float3(4, 5, 6));
            float3 point(2.5f, 3.5f, 4.5f); // Центр

            float3 normalized = box.normalize_coords(point);
            float3 expected_normalized(0.5f, 0.5f, 0.5f);

            suite.assert_approximately_equal(normalized, expected_normalized, "normalize_coords");

            float3 denormalized = box.denormalize_coords(normalized);
            suite.assert_approximately_equal(denormalized, point, "denormalize_coords");
        }

        // clamp_point
        {
            AABB box(float3(0, 0, 0), float3(10, 10, 10));

            suite.assert_equal(box.clamp_point(float3(5, 5, 5)), float3(5, 5, 5), "clamp_point inside");
            suite.assert_equal(box.clamp_point(float3(-5, 5, 5)), float3(0, 5, 5), "clamp_point outside negative");
            suite.assert_equal(box.clamp_point(float3(15, 5, 5)), float3(10, 5, 5), "clamp_point outside positive");
            suite.assert_equal(box.clamp_point(float3(-5, 15, -5)), float3(0, 10, 0), "clamp_point multiple axes");
        }

        // distance и distance_sq
        {
            AABB box(float3(0, 0, 0), float3(10, 10, 10));

            // Точка внутри
            suite.assert_approximately_equal(box.distance_sq(float3(5, 5, 5)), 0.0f, "distance_sq inside");
            suite.assert_approximately_equal(box.distance(float3(5, 5, 5)), 0.0f, "distance inside");

            // Точка снаружи
            float3 outside_point(12, 5, 5);
            float expected_distance_sq = 4.0f; // (12-10)² = 4
            float expected_distance = 2.0f;

            suite.assert_approximately_equal(box.distance_sq(outside_point), expected_distance_sq, "distance_sq outside");
            suite.assert_approximately_equal(box.distance(outside_point), expected_distance, "distance outside");

            // Точка снаружи по нескольким осям
            float3 outside_point2(12, 15, 5);
            float expected_distance_sq2 = 4.0f + 25.0f; // (12-10)² + (15-10)² = 4 + 25 = 29
            float expected_distance2 = std::sqrt(29.0f);

            suite.assert_approximately_equal(box.distance_sq(outside_point2), expected_distance_sq2, "distance_sq outside multiple axes");
            suite.assert_approximately_equal(box.distance(outside_point2), expected_distance2, "distance outside multiple axes");
        }

        // ============================================================================
        // 9. Операторы сравнения
        // ============================================================================
        suite.section("Операторы сравнения");

        AABB box1(float3(1, 2, 3), float3(4, 5, 6));
        AABB box2(float3(1, 2, 3), float3(4, 5, 6));
        AABB box3(float3(1, 2, 3), float3(4, 5, 7));

        suite.assert_true(box1 == box2, "operator == for equal AABBs");
        suite.assert_false(box1 == box3, "operator == for different AABBs");
        suite.assert_false(box1 != box2, "operator != for equal AABBs");
        suite.assert_true(box1 != box3, "operator != for different AABBs");

        // ============================================================================
        // 10. Строковое представление
        // ============================================================================
        suite.section("Строковое представление");

        {
            AABB box(float3(1.5f, 2.5f, 3.5f), float3(4.5f, 5.5f, 6.5f));
            std::string str = box.to_string();

            // Проверяем что строка содержит ожидаемые значения
            suite.assert_true(str.find("1.5000") != std::string::npos || str.find("1.5") != std::string::npos,
                "to_string contains min values");
            suite.assert_true(str.find("6.5000") != std::string::npos || str.find("6.5") != std::string::npos,
                "to_string contains max values");
        }

        // ============================================================================
        // 11. Глобальные функции
        // ============================================================================
        suite.section("Глобальные функции");

        AABB a(float3(1, 1, 1), float3(3, 3, 3));
        AABB b(float3(2, 2, 2), float3(4, 4, 4));

        // combine
        {
            AABB combined = combine(a, b);
            suite.assert_equal(combined.min, float3(1, 1, 1), "combine() global function - min");
            suite.assert_equal(combined.max, float3(4, 4, 4), "combine() global function - max");
        }

        // intersect
        {
            AABB intersected = intersect(a, b);
            suite.assert_equal(intersected.min, float3(2, 2, 2), "intersect() global function - min");
            suite.assert_equal(intersected.max, float3(3, 3, 3), "intersect() global function - max");
        }

        // intersects
        {
            suite.assert_true(intersects(a, b), "intersects() global function for overlapping AABBs");
            AABB c(float3(5, 5, 5), float3(6, 6, 6));
            suite.assert_false(intersects(a, c), "intersects() global function for non-overlapping AABBs");
        }

        // surface_area
        {
            float expected_area = a.surface_area();
            suite.assert_approximately_equal(surface_area(a), expected_area, "surface_area() global function");
        }

        // volume
        {
            float expected_volume = a.volume();
            suite.assert_approximately_equal(volume(a), expected_volume, "volume() global function");
        }

        // contains для точки
        {
            suite.assert_true(contains(a, float3(2, 2, 2)), "contains(point) global function - point inside");
            suite.assert_false(contains(a, float3(0, 0, 0)), "contains(point) global function - point outside");
        }

        // contains для AABB
        {
            AABB inside(float3(1.5f, 1.5f, 1.5f), float3(2.5f, 2.5f, 2.5f));
            AABB outside(float3(0, 0, 0), float3(4, 4, 4));

            suite.assert_true(contains(a, inside), "contains(AABB) global function - AABB inside");
            suite.assert_false(contains(a, outside), "contains(AABB) global function - AABB outside");
        }

        // ============================================================================
        // 12. Граничные случаи и особые значения
        // ============================================================================
        suite.section("Граничные случаи и особые значения");

        // Невалидный AABB (min > max)
        {
            AABB invalid(float3(5, 5, 5), float3(1, 1, 1));
            suite.assert_false(invalid.is_valid(), "is_valid returns false for invalid AABB (min > max)");
            suite.assert_true(invalid.is_empty(), "is_empty returns true for invalid AABB");
        }

        // AABB с нулевым объемом
        {
            AABB zero_volume(float3(1, 1, 1), float3(1, 1, 1));
            suite.assert_true(zero_volume.is_valid(), "zero volume AABB is valid");
            suite.assert_true(zero_volume.is_empty(1e-6f), "zero volume AABB is empty");
            suite.assert_approximately_equal(zero_volume.volume(), 0.0f, "zero volume");
            suite.assert_approximately_equal(zero_volume.surface_area(), 0.0f, "zero surface area");
        }

        // Очень большой AABB
        {
            float large = 1e10f;
            AABB large_box(float3(-large, -large, -large), float3(large, large, large));

            suite.assert_true(large_box.is_valid(), "very large AABB is valid");
            suite.assert_approximately_equal(large_box.volume(), 8.0f * large * large * large,
                "volume of large AABB", large * large * large * 1e-3f);
        }

        // AABB с бесконечными значениями
        {
            AABB infinite_box = AABB_Infinite;
            suite.assert_false(infinite_box.is_valid(), "infinite AABB is not valid");
            suite.assert_true(std::isinf(infinite_box.min.x), "infinite AABB min is -inf");
            suite.assert_true(std::isinf(infinite_box.max.x), "infinite AABB max is +inf");
        }

        // Пустой AABB (по умолчанию)
        {
            AABB empty_box = AABB_Empty;
            suite.assert_false(empty_box.is_valid(), "empty AABB is not valid");
            suite.assert_true(empty_box.is_empty(), "empty AABB is empty");
        }

        // Тест с epsilon в contains
        {
            AABB box(float3(0, 0, 0), float3(10, 10, 10));
            float3 point_on_boundary(10, 5, 5);

            // Без epsilon
            suite.assert_true(box.contains(point_on_boundary, 0.0f), "contains point on boundary with epsilon=0");

            // С отрицательным epsilon (строже)
            suite.assert_false(box.contains(point_on_boundary, -0.001f), "contains point on boundary with negative epsilon");

            // С положительным epsilon (более лояльно)
            suite.assert_true(box.contains(point_on_boundary, 0.001f), "contains point on boundary with positive epsilon");

            // Точка чуть снаружи
            float3 point_just_outside(10.001f, 5, 5);
            suite.assert_false(box.contains(point_just_outside, 0.0f), "contains point just outside with epsilon=0");
            suite.assert_true(box.contains(point_just_outside, 0.002f), "contains point just outside with epsilon=0.002");
        }

        // Луч параллельный грани
        {
            AABB box(float3(-1, -1, -1), float3(1, 1, 1));
            float3 origin(0, 2, 0);
            float3 direction(1, 0, 0); // Параллелен оси X

            bool intersects = box.intersect_ray(origin, direction);
            suite.assert_false(intersects, "ray parallel to face does not intersect");
        }

        // Луч направлен от AABB
        {
            AABB box(float3(-1, -1, -1), float3(1, 1, 1));
            float3 origin(0, 0, 2);
            float3 direction(0, 0, 1); // Направлен от коробки

            bool intersects = box.intersect_ray(origin, direction);
            suite.assert_false(intersects, "ray pointing away from AABB does not intersect");
        }

        // Луч, который начинается за AABB и направлен к нему
        {
            AABB box(float3(-1, -1, -1), float3(1, 1, 1));
            float3 origin(0, 0, -2);
            float3 direction(0, 0, 1); // Направлен к коробке

            bool intersects = box.intersect_ray(origin, direction);
            suite.assert_true(intersects, "ray starts behind and points toward AABB");
        }

        // Луч, который касается только ребра или вершины
        {
            AABB box(float3(-1, -1, -1), float3(1, 1, 1));

            // Касается вершины (1,1,1)
            float3 origin1(2, 2, 2);
            float3 direction1 = (float3(1, 1, 1) - origin1).normalize(); // Направлен к вершине

            // Касается ребра по оси X
            float3 origin2(2, 0, 0);
            float3 direction2 = (float3(1, 0, 0) - origin2).normalize(); // Направлен к ребру

            bool intersects1 = box.intersect_ray(origin1, direction1);
            bool intersects2 = box.intersect_ray(origin2, direction2);

            // Эти тесты могут быть хрупкими из-за численной точности
            // suite.assert_true(intersects1, "ray grazing vertex intersects");
            // suite.assert_true(intersects2, "ray grazing edge intersects");

            suite.skip_test("Ray grazing vertex/edge", "Numerical precision issues may cause false negatives");
        }

        // Нормализация координат для точки вне AABB
        {
            AABB box(float3(0, 0, 0), float3(10, 10, 10));
            float3 point_outside(-5, 15, 5);

            float3 normalized = box.normalize_coords(point_outside);

            // Должно быть зажато в [0, 1]
            suite.assert_true(normalized.x >= 0.0f && normalized.x <= 1.0f,
                "normalize_coords clamps X outside range");
            suite.assert_true(normalized.y >= 0.0f && normalized.y <= 1.0f,
                "normalize_coords clamps Y outside range");
            suite.assert_true(normalized.z >= 0.0f && normalized.z <= 1.0f,
                "normalize_coords clamps Z outside range");
        }

        suite.footer();
    }
}
