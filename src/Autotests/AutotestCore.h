// Author: DeepSeek, NSDeathman
#pragma once

#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <iomanip>
#include <chrono>
#include <sstream>
#include <limits>

// Include the math library
#include "../MathAPI/MathAPI.h"

namespace MathTests
{
    using namespace Math;

    // ============================================================================
    // Test Configuration
    // ============================================================================

    struct TestConfig
    {
        static constexpr bool VERBOSE_OUTPUT = true;
        static constexpr int PRECISION = 6;
        static constexpr float DEFAULT_EPSILON = 1e-5f;
        static constexpr float STRICT_EPSILON = 1e-7f;
    };

    // ============================================================================
    // Test Utilities
    // ============================================================================

    class TestReporter
    {
    private:
        std::ostream& output;
        int current_indent;

    public:
        TestReporter(std::ostream& os = std::cout) : output(os), current_indent(0) {}

        void indent() { current_indent += 2; }
        void unindent() { current_indent = std::max(0, current_indent - 2); }

        void message(const std::string& msg)
        {
            output << std::string(current_indent, ' ') << msg << std::endl;
        }

        void warning(const std::string& msg)
        {
            output << std::string(current_indent, ' ') << "WARNING: " << msg << std::endl;
        }

        void error(const std::string& msg)
        {
            output << std::string(current_indent, ' ') << "ERROR: " << msg << std::endl;
        }

        void section(const std::string& title)
        {
            output << "\n" << std::string(current_indent, ' ')
                << "=== " << title << " ===" << std::endl;
        }
    };

    // Helper function for safe approximately comparison
    template<typename T>
    bool safe_approximately(const T& a, const T& b, float epsilon)
    {
        // For arithmetic types
        if constexpr (std::is_arithmetic_v<T>) {
            if (std::isinf(a) && std::isinf(b)) {
                return std::signbit(a) == std::signbit(b);
            }
            if (std::isnan(a) && std::isnan(b)) {
                return true;
            }
            if (std::isinf(a) || std::isinf(b) || std::isnan(a) || std::isnan(b)) {
                return false;
            }
            return std::abs(a - b) <= epsilon;
        }
        // For vector/matrix types with approximately method
        else if constexpr (requires { a.approximately(b, epsilon); }) {
            return a.approximately(b, epsilon);
        }
        // For other types (should have operator==)
        else {
            return a == b;
        }
    }

    class TestSuite
    {
    private:
        std::string suite_name;
        int tests_passed;
        int tests_failed;
        int tests_skipped;
        bool verbose;
        TestReporter reporter;
        std::chrono::steady_clock::time_point start_time;

        struct TestResult
        {
            std::string name;
            bool passed;
            std::string message;
            double duration_ms;
        };

        std::vector<TestResult> test_results;

    public:
        TestSuite(const std::string& name, bool verb = TestConfig::VERBOSE_OUTPUT)
            : suite_name(name), tests_passed(0), tests_failed(0), tests_skipped(0),
            verbose(verb), reporter() {}

        // Section management
        void section(const std::string& title)
        {
            reporter.section(title);
        }

        // Test lifecycle management
        void start_test(const std::string& test_name)
        {
            if (verbose)
            {
                reporter.message("RUN: " + test_name);
                reporter.indent();
            }
            start_time = std::chrono::steady_clock::now();
        }

        void end_test(const std::string& test_name, bool passed, const std::string& message = "")
        {
            auto end_time = std::chrono::steady_clock::now();
            double duration = std::chrono::duration<double, std::milli>(end_time - start_time).count();

            if (verbose) reporter.unindent();

            TestResult result{ test_name, passed, message, duration };
            test_results.push_back(result);

            if (passed)
            {
                tests_passed++;
                if (verbose)
                {
                    reporter.message("+ PASS: " + test_name + " (" + std::to_string(duration) + "ms)");
                }
            }
            else
            {
                tests_failed++;
                reporter.error("- FAIL: " + test_name + " - " + message + " (" + std::to_string(duration) + "ms)");
            }
        }

        void skip_test(const std::string& test_name, const std::string& reason = "")
        {
            tests_skipped++;
            reporter.message("- SKIP: " + test_name + (reason.empty() ? "" : " - " + reason));
        }

        // Value formatting for error messages
        template<typename T>
        std::string format_value(const T& value)
        {
            std::ostringstream oss;
            oss << std::setprecision(TestConfig::PRECISION);

            if constexpr (std::is_arithmetic_v<T>)
            {
                if (std::isinf(value)) {
                    oss << (value < 0 ? "-inf" : "inf");
                }
                else if (std::isnan(value)) {
                    oss << "nan";
                }
                else {
                    oss << value;
                }
            }
            else if constexpr (requires { value.to_string(); })
            {
                oss << value.to_string();
            }
            else if constexpr (requires { oss << value; })
            {
                oss << value;
            }
            else
            {
                oss << "[unprintable type]";
            }
            return oss.str();
        }

        // Assertion methods
        template<typename T>
        bool assert_equal(const T& actual, const T& expected, const std::string& test_name,
            float epsilon = TestConfig::DEFAULT_EPSILON)
        {
            start_test(test_name);

            bool success = safe_approximately(actual, expected, epsilon);
            std::string message;

            if (!success)
            {
                message = "Expected: " + format_value(expected) +
                    ", Got: " + format_value(actual) +
                    ", Epsilon: " + std::to_string(epsilon);
            }

            end_test(test_name, success, message);
            return success;
        }

        // Alias for assert_equal with more descriptive name for floating-point comparisons
        template<typename T>
        bool assert_approximately_equal(const T& actual, const T& expected, const std::string& test_name,
            float epsilon = TestConfig::DEFAULT_EPSILON)
        {
            return assert_equal(actual, expected, test_name, epsilon);
        }

        template<typename T>
        bool assert_not_equal(const T& actual, const T& expected, const std::string& test_name,
            float epsilon = TestConfig::DEFAULT_EPSILON)
        {
            start_test(test_name);

            bool success = !safe_approximately(actual, expected, epsilon);
            std::string message;

            if (!success)
            {
                message = "Values should not be equal: " + format_value(actual);
            }

            end_test(test_name, success, message);
            return success;
        }

        template<typename T>
        bool assert_true(const T& condition, const std::string& test_name)
        {
            start_test(test_name);

            bool success = static_cast<bool>(condition);
            std::string message = success ? "" : "Condition evaluated to false";

            end_test(test_name, success, message);
            return success;
        }

        template<typename T>
        bool assert_false(const T& condition, const std::string& test_name)
        {
            start_test(test_name);

            bool success = !static_cast<bool>(condition);
            std::string message = success ? "" : "Condition evaluated to true";

            end_test(test_name, success, message);
            return success;
        }

        template<typename T>
        bool assert_nan(const T& value, const std::string& test_name)
        {
            start_test(test_name);

            bool success = std::isnan(static_cast<float>(value));
            std::string message = success ? "" : "Value is not NaN";

            end_test(test_name, success, message);
            return success;
        }

        template<typename T>
        bool assert_infinity(const T& value, const std::string& test_name)
        {
            start_test(test_name);

            bool success = std::isinf(static_cast<float>(value));
            std::string message = success ? "" : "Value is not infinity";

            end_test(test_name, success, message);
            return success;
        }

        template<typename T>
        bool assert_finite(const T& value, const std::string& test_name)
        {
            start_test(test_name);

            bool success = std::isfinite(static_cast<float>(value));
            std::string message = success ? "" : "Value is not finite";

            end_test(test_name, success, message);
            return success;
        }

        // Test suite management
        void header()
        {
            std::cout << "\n" << std::string(70, '=') << std::endl;
            std::cout << "TEST SUITE: " << suite_name << std::endl;
            std::cout << std::string(70, '=') << std::endl;
            test_results.clear();
        }

        void footer()
        {
            std::cout << std::string(70, '-') << std::endl;
            std::cout << "RESULTS: " << tests_passed << " passed, "
                << tests_failed << " failed, "
                << tests_skipped << " skipped" << std::endl;

            if (tests_failed > 0 && verbose)
            {
                std::cout << "\nFAILED TESTS:" << std::endl;
                for (const auto& result : test_results)
                {
                    if (!result.passed)
                    {
                        std::cout << "  - " << result.name << ": " << result.message << std::endl;
                    }
                }
            }

            std::cout << std::string(70, '=') << std::endl;
        }

        // Statistics
        int get_total_count() const { return tests_passed + tests_failed + tests_skipped; }
        int get_passed_count() const { return tests_passed; }
        int get_failed_count() const { return tests_failed; }
        int get_skipped_count() const { return tests_skipped; }
        double get_success_rate() const
        {
            int total = get_total_count();
            return total > 0 ? (static_cast<double>(tests_passed) / total) * 100.0 : 0.0;
        }
    };
} // namespace MathTests
