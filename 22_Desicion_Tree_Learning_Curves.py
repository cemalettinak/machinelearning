


def basarimgrafigi_hesapla(algo, X_egitim, X_test, y_egitim, y_test):

    egitim_dogruluk = []
    test_dogruluk = []
    
    for i in range(1, len(X_egitim) + 1):
    
        algo.fit(X_egitim[:i], y_egitim[:i])
        
        y_egitim_tahmin = algo.predict(X_egitim[:i])
        egitim_dogruluk.append(accuracy_score(y_egitim[:i], y_egitim_tahmin))
        y_test_tahmin = algo.predict(X_test)
        test_dogruluk.append(accuracy_score(y_test, y_test_tahmin))
        
    plt.plot([i for i in range(1, len(X_egitim) + 1)],
             egitim_dogruluk, label="Eğitim")
    plt.plot([i for i in range(1, len(X_egitim) + 1)],
             test_dogruluk, label="Test")
             
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Learning Curves')
    plt.legend()
    plt.show()
    
basarimgrafigi_hesapla(dt_model, X_egitim, X_test, y_egitim, y_test)  
