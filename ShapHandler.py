import shap
import matplotlib.pyplot as plt

shap.initjs()


def check_importance(X_train, Y_train, X_test, model, Y_type, name, featureset, fold):
    model.fit(X_train, Y_train)
    explainer = shap.KernelExplainer(model.predict_proba, X_train)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    # plt.legend('')
    plt.xlabel('Importance')
    plt.savefig('./feature figs/' + Y_type + '_' + name + '_' + str(fold) + '_' + featureset + '.jpg', bbox_inches='tight', format='jpg')
    plt.clf()
    # plt.savefig(name + '.jpg')
