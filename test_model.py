from model import train_and_predict

def test_predictions_not_none():
    """
    Test 1: Sprawdza, czy otrzymujemy jakąkolwiek predykcję.
    """
    preds, _, _ = train_and_predict()
    assert preds is not None, "Predictions should not be None."

def test_predictions_length():
    """
    Test 2: Sprawdza, czy długość listy predykcji jest większa od 0 i czy odpowiada przewidywanej liczbie próbek testowych.
    """
    preds, _ , test_len = train_and_predict()
    assert len(preds) > 0, "Predictions list should not be empty."
    assert len(preds) == test_len, "Predictions length should match the number of test samples."

def test_predictions_value_range():
    """
    Test 3: Sprawdza, czy wartości w predykcjach mieszczą się w spodziewanym zakresie: Dla zbioru Iris mamy 3 klasy (0, 1, 2).
    """
    preds, _ , _ = train_and_predict()
    assert all(0 <= pred <= 2 for pred in preds), "All predictions should be in the range [0, 2]."

def test_model_accuracy():
    """
    Test 4: Sprawdza, czy model osiąga co najmniej 70% dokładności.
    """
    _, accuracy, _ = train_and_predict()
    assert accuracy >= 0.7, "Model accuracy should be at least 70%."