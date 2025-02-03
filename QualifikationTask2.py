import unittest
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import stats


class TUStudent:
    """ "Represents all the relevant data of a TU-Darmstadt student"""

    def __init__(
        self, name, age, date_of_registration, study_program, registration_number
    ):
        self.name = name
        self.age = age
        self.date_of_registration = date_of_registration
        self.study_program = study_program
        self.registration_number = registration_number

    # Getters
    def get_name(self):
        return self.name

    def get_age(self):
        return self.age

    def get_date_of_registration(self):
        return self.date_of_registration

    def get_study_program(self):
        return self.study_program

    def get_registration_number(self):
        return self.registration_number

    # Setters
    def set_name(self, name):
        self.name = name

    def set_age(self, age):
        self.age = age

    def set_date_of_registration(self, date_of_registration):
        self.date_of_registration = date_of_registration

    def set_study_program(self, study_program):
        self.study_program = study_program

    def set_registration_number(self, registration_number):
        self.registration_number = registration_number


class Robust_Data_Science_Student(TUStudent):
    """ "contains relevant Data Sience methods"""

    def solve_integral_problem(self, x_range, x_stats, plot_derivative=False):

        # valaidate inputs
        if not isinstance(x_range, (np.ndarray, list)) or not isinstance(
            x_stats, (np.ndarray, list)
        ):
            raise ValueError("wrong inputtype")

        x = np.linspace(x_range[0], x_range[1], 1000)
        y = np.exp(-x) * np.cos(x)

        # Compute statistics
        y_stats = y[
            (x >= x_stats[0]) & (x <= x_stats[1])
        ]  # boolean mask ignores y values out of the statistic intervall

    mean_y = np.mean(y_stats)
    var_y = np.var(y_stats)
    std_y = np.std(y_stats)

    # 70% threshold
    y_m = scoreatpercentile(y_stats, 70)

    # Compute derivative
    dy_dx = np.gradient(y, x)

    # find index where dy/dx = 0 with small tolerance
    zero_derivative_indices = np.where(np.isclose(dy_dx, 0, atol=1e-5))[0]

    def solve_linear_algebra(A, B):
        # Validate input types
        if not isinstance(A, (list, np.ndarray)) or not isinstance(
            B, (list, np.ndarray)
        ):
            raise TypeError("A and B should be numpy array or list")

        # Prevent type errors
        A = np.array(A, dtype=float)
        B = np.array(B, dtype=float)

        # Check if A is a square matrix
        rows, cols = A.shape
        if rows != cols:
            raise ValueError("Matrix A must be square")

        # Check if B has the correct dimensions
        if B.shape[0] != rows:
            raise ValueError("Size of vector B should be same as rows in A.")

        # Solve for x
        try:
            solution = np.linalg.solve(A, B)
            return solution

        except np.linalg.LinAlgError:
            raise ValueError("The system does not have a unique solution.")


# Unit Tests
class TestTUStudent(unittest.TestCase):
    def setUp(self):
        """Set up a test student object before each test"""
        self.student = TUStudent(
            name="Max Mustermann",
            age=21,
            date_of_registration="2023-01-01",
            study_program="Data Science",
            registration_number="123456",
        )

    def test_getters(self):
        """Test getter methods"""
        self.assertEqual(self.student.get_name(), "Max Mustermann")
        self.assertEqual(self.student.get_age(), 21)
        self.assertEqual(self.student.get_date_of_registration(), "2023-01-01")
        self.assertEqual(self.student.get_study_program(), "Data Science")
        self.assertEqual(self.student.get_registration_number(), "123456")

    def test_setters(self):
        """Test setter methods"""
        self.student.set_name("Max")
        self.assertEqual(self.student.get_name(), "Max")

        self.student.set_age(21)
        self.assertEqual(self.student.get_age(), 21)

        self.student.set_date_of_registration("2001-01-01")
        self.assertEqual(self.student.get_date_of_registration(), "2001-01-01")

        self.student.set_study_program("Engineering")
        self.assertEqual(self.student.get_study_program(), "Engineering")

        self.student.set_registration_number("656901")
        self.assertEqual(self.student.get_registration_number(), "656901")

    def test_invalid_inputs(self):
        """Optional: Test invalid inputs (e.g., empty strings or negative ages)"""
        with self.assertRaises(ValueError):
            self.student.set_age(-1)

        with self.assertRaises(ValueError):
            self.student.set_name("")


def main():



if __name__ == "__main__":
    main()
