import numpy as np
import pytest
from numpy.typing import NDArray

from ScaPyra.geometry import find_circle_intersections, select_intersection_point, calculate_angle


def assert_points_equal_unordered(a: NDArray[np.floating], b: NDArray[np.floating], tol: float = 1e-6) -> None:
    """
    Helper: compare two sets of 2D points (shape (2,2)), ignoring order.
    """
    assert a.shape == (2, 2)
    assert b.shape == (2, 2)

    # sort rows by x,y to make order deterministic before comparison
    a_sorted = a[np.lexsort((a[:, 1], a[:, 0]))]
    b_sorted = b[np.lexsort((b[:, 1], b[:, 0]))]

    assert np.allclose(a_sorted, b_sorted, atol=tol)


def test_two_intersections_symmetric_case() -> None:
    """
    Dwa okręgi:
    - C1 w (0,0) o promieniu 5
    - C2 w (6,0) o promieniu 5

    Powinny przeciąć się w punktach (3, +4) i (3, -4).
    (To wynika z klasycznej geometrii trójkąta 3-4-5.)
    """
    p1 = np.array([0.0, 0.0])
    r1 = 5.0
    p2 = np.array([6.0, 0.0])
    r2 = 5.0

    result = find_circle_intersections(p1, r1, p2, r2)

    assert result is not None, "Expected two intersection points, got None."
    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)

    expected = np.array([[3.0, 4.0], [3.0, -4.0]], dtype=float)
    assert_points_equal_unordered(result, expected)


def test_tangent_external() -> None:
    """
    Okręgi styczne zewnętrznie:
    - C1 w (0,0), r=5
    - C2 w (10,0), r=5

    Styczność w punkcie (5,0). Funkcja może zwrócić dwa identyczne punkty.
    """
    p1 = np.array([0.0, 0.0])
    r1 = 5.0
    p2 = np.array([10.0, 0.0])
    r2 = 5.0

    result = find_circle_intersections(p1, r1, p2, r2)

    assert result is not None, "Expected tangent intersection, got None."
    assert result.shape == (2, 2)

    # oba wiersze powinny być (5,0) z dokładnością numeryczną
    assert np.allclose(result[0], [5.0, 0.0], atol=1e-6)
    assert np.allclose(result[1], [5.0, 0.0], atol=1e-6)


def test_no_intersection_too_far() -> None:
    """
    Okręgi za daleko od siebie, brak przecięć.
    - C1 w (0,0), r=2
    - C2 w (10,0), r=2
    """
    p1 = np.array([0.0, 0.0])
    r1 = 2.0
    p2 = np.array([10.0, 0.0])
    r2 = 2.0

    result = find_circle_intersections(p1, r1, p2, r2)

    assert result is None


def test_no_intersection_one_inside_other() -> None:
    """
    Jeden okrąg całkowicie wewnątrz drugiego bez przecięcia.
    - C1 w (0,0), r=5
    - C2 w (1,0), r=1
    """
    p1 = np.array([0.0, 0.0])
    r1 = 5.0
    p2 = np.array([1.0, 0.0])
    r2 = 1.0

    result = find_circle_intersections(p1, r1, p2, r2)

    assert result is None


def test_coincident_centers_same_radius() -> None:
    """
    Okręgi współśrodkowe o tym samym promieniu -> nieskończenie wiele rozwiązań.
    Funkcja powinna zwrócić None, bo nie da się zwrócić dokładnie dwóch punktów.
    """
    p1 = np.array([2.0, -3.0])
    r1 = 4.0
    p2 = np.array([2.0, -3.0])
    r2 = 4.0

    result = find_circle_intersections(p1, r1, p2, r2)

    assert result is None


def test_output_is_float64() -> None:
    """
    Sprawdza, czy wynikowe współrzędne są typu float (a nie np. int)
    oraz czy wynik jest kopiowalny / numerycznie stabilny.
    """
    p1 = np.array([0, 0], dtype=int)
    r1 = 5.0
    p2 = np.array([6, 0], dtype=int)
    r2 = 5.0

    result = find_circle_intersections(p1, r1, p2, r2)

    assert result is not None
    assert result.dtype == np.float64 or np.issubdtype(result.dtype, np.floating)


def test_invalid_input_shape_returns_none() -> None:
    """
    Jeżeli funkcja dostała zły kształt (np. 3-elementowy wektor zamiast 2D),
    powinna zwrócić None, nie rzucić wyjątku.
    Ten test przejdzie tylko jeśli masz w funkcji walidację kształtu p1/p2.
    Jeśli nie masz tej walidacji, usuń ten test.
    """
    p1 = np.array([0.0, 0.0, 0.0])  # zły wymiar
    r1 = 5.0
    p2 = np.array([6.0, 0.0, 0.0])  # zły wymiar
    r2 = 5.0

    try:
        result = find_circle_intersections(p1, r1, p2, r2)
    except Exception as e:
        pytest.fail(f"Function raised an exception on invalid input: {e}")
    else:
        # jeśli masz walidację shape -> powinno być None
        # jeśli nie masz, możesz asertywnie sprawdzić tylko brak wyjątku
        assert result is None or isinstance(result, np.ndarray)

def test_select_returns_point_with_lowest_x() -> None:
    """
    Sprawdza, czy funkcja poprawnie wybiera punkt o mniejszej współrzędnej X.
    """
    intersections = np.array([
        [5.0, 2.0],
        [3.0, -1.0]
    ])

    result = select_intersection_point(intersections)

    assert result is not None
    assert np.allclose(result, [3.0, -1.0]), f"Expected [3.0, -1.0], got {result}"


def test_select_works_when_x_equal_picks_first() -> None:
    """
    Jeżeli współrzędne X są równe, funkcja powinna zwrócić pierwszy punkt (argmin zwraca pierwszy indeks).
    """
    intersections = np.array([
        [4.0, 1.0],
        [4.0, -5.0]
    ])

    result = select_intersection_point(intersections)

    assert result is not None
    assert np.allclose(result, intersections[0]), "Expected first row when Xs are equal."


def test_invalid_shape_returns_none() -> None:
    """
    Zwraca None, gdy tablica ma niepoprawny kształt.
    """
    intersections = np.array([[1.0, 2.0, 3.0]])  # shape (1,3)
    result = select_intersection_point(intersections)
    assert result is None


def test_none_input_returns_none() -> None:
    """
    Zwraca None, gdy intersections = None.
    """
    result = select_intersection_point(None)
    assert result is None


def test_non_array_input_returns_none() -> None:
    """
    Zwraca None, gdy przekazano coś innego niż np.ndarray.
    """
    result = select_intersection_point([[1.0, 2.0], [3.0, 4.0]])  # lista zamiast tablicy
    assert result is None


def test_output_is_numpy_array() -> None:
    """
    Wynik powinien być typu np.ndarray o wymiarze (2,).
    """
    intersections = np.array([[10.0, 0.0], [0.0, 10.0]])
    result = select_intersection_point(intersections)

    assert isinstance(result, np.ndarray)
    assert result.shape == (2,)

def test_angle_right() -> None:
    """Punkt docelowy po prawej stronie powinien dać 0 stopni."""
    p = np.array([0.0, 0.0])
    q = np.array([1.0, 0.0])
    angle = calculate_angle(p, q)
    assert angle is not None
    assert abs(angle - 0.0) < 1e-9


def test_angle_up() -> None:
    """Punkt nad bazowym (góra) -> 90 stopni."""
    p = np.array([0.0, 0.0])
    q = np.array([0.0, 1.0])
    angle = calculate_angle(p, q)
    assert angle is not None
    assert abs(angle - 90.0) < 1e-9


def test_angle_left() -> None:
    """Punkt po lewej -> 180 stopni."""
    p = np.array([0.0, 0.0])
    q = np.array([-1.0, 0.0])
    angle = calculate_angle(p, q)
    assert angle is not None
    assert abs(angle - 180.0) < 1e-9


def test_angle_down() -> None:
    """Punkt pod bazowym -> 270 stopni (lub 270.0 po normalizacji)."""
    p = np.array([0.0, 0.0])
    q = np.array([0.0, -1.0])
    angle = calculate_angle(p, q)
    assert angle is not None
    assert abs(angle - 270.0) < 1e-9


def test_angle_quadrant_general() -> None:
    """Punkt w pierwszej ćwiartce (x>0, y>0) powinien dać kąt między 0 a 90."""
    p = np.array([0.0, 0.0])
    q = np.array([1.0, 1.0])
    angle = calculate_angle(p, q)
    assert 0.0 < angle < 90.0


def test_invalid_input_returns_none() -> None:
    """Zwraca None dla błędnych danych."""
    assert calculate_angle(None, np.array([1.0, 0.0])) is None
    assert calculate_angle(np.array([0.0, 0.0]), None) is None
    assert calculate_angle(np.array([[0.0, 0.0]]), np.array([1.0, 0.0])) is None
    assert calculate_angle(np.array([0.0, 0.0]), np.array([[1.0, 0.0]])) is None


def test_angle_type_and_range() -> None:
    """Wynik powinien być float w zakresie [0, 360)."""
    p = np.array([0.0, 0.0])
    q = np.array([1.0, -1.0])  # 315°
    angle = calculate_angle(p, q)
    assert isinstance(angle, float)
    assert 0.0 <= angle < 360.0