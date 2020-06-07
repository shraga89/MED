import shap
import matplotlib.pyplot as plt
import xgboost

shap.initjs()


def check_importance(X_train, Y_train, X_test, model, Y_type, name, featureset, fold):
    # print("Not exporting plots")
    # return None
    f = plt.figure()
    model.fit(X_train, Y_train)
    # model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X_train, label=Y_train), 100)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test)
    # plt.legend('')
    plt.xlabel('Importance')
    f.savefig('./feature figs new/' + Y_type + '_' + name + '_' + str(fold) + '_' + featureset + '.jpg', bbox_inches='tight', dpi=600)
    plt.clf()
    # plt.savefig(name + '.jpg')


# def check_importance(X_train, Y_train, X_test, model, Y_type, name, featureset, fold):
#     print("Not exporting plots")
#     return None
#     model.fit(X_train, Y_train)
#     explainer = shap.KernelExplainer(model.predict_proba, X_train)
#     shap_values = explainer.shap_values(X_test)
#     shap.summary_plot(shap_values, X_test)
#     # plt.legend('')
#     plt.xlabel('Importance')
#     plt.savefig('./feature figs/' + Y_type + '_' + name + '_' + str(fold) + '_' + featureset + '.jpg', bbox_inches='tight', format='jpg')
#     plt.clf()
#     plt.savefig(name + '.jpg')
