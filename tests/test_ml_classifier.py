from ml_classifier import load_train_test, predict, train


def test_ml_classifier():
    train_data, _ = load_train_test()
    model = train(train_data)
    assert predict(model, "WINNER! You won a FREE prize. Call 09061701461 to claim") == "spam"
    assert predict(model, "Ok, see you at 6 for dinner") == "ham"
