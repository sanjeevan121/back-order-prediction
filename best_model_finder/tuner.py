from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics  import roc_auc_score,accuracy_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB


class Model_Finder:
    """
                This class shall  be used to find the model with best accuracy and AUC score.
                Written By: Sanjeevan Thorat
                Version: 1.0
                Revisions: None

                """

    def __init__(self,file_object,logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.clf = RandomForestClassifier()
        self.xgb = XGBClassifier(objective='binary:logistic')
        self.svm=SVC()
        self.logistic=LogisticRegressionCV()
        self.catboost=CatBoostClassifier()
        self.nb=GaussianNB()


    def get_best_params_for_logistic_regression(self,train_x,train_y):
        """
                                Method Name: get_best_params_logistic_regression
                                Description: get the parameters for logistic regression Algorithm which give the best accuracy.
                                             Use Hyper Parameter Tuning.
                                Output: The model with the best parameters
                                On Failure: Raise Exception

                                Written By: Sanjeevan Thorat
                                Version: 1.0
                                Revisions: None

                        """
        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_logistic_regression method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {'Cs':[0.1,0.3,1,3,10],
                               'cv':[3,5,10],
                               'solver':['newton-cg','lbfgs','sag']
                               }

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.logistic, param_grid=self.param_grid, verbose=3)
            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #extracting the best parameters
            self.Cs = self.grid.best_params_['Cs']
            self.cv = self.grid.best_params_['cv']
            self.solver = self.grid.best_params_['solver']


            #creating a new model with the best parameters
            self.logistic = LogisticRegressionCV(Cs=self.Cs, cv=self.cv, solver=self.solver)
            # training the mew model
            self.logistic.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'Logistic Regression best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_logistic_regression method of the Model_Finder class')

            return self.logistic
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_logistic_regression method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Logistic Regression Parameter tuning  failed. Exited the get_best_params_for_logistic_regression method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_svm(self,train_x,train_y):
        """
                                Method Name: get_best_params_for_svm
                                Description: get the parameters for svm Algorithm which give the best accuracy.
                                             Use Hyper Parameter Tuning.
                                Output: The model with the best parameters
                                On Failure: Raise Exception

                                Written By: Sanjeevan Thorat
                                Version: 1.0
                                Revisions: None

                        """
        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_svm of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {"kernel": ['linear','poly','rbf','sigmoid'], "gamma": ['scale','auto'],
                               "C" :[0.1,0.3,1,3,10,30,100],'degree':[1,2,3,4,5,6]}
            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.svm, param_grid=self.param_grid, cv=5,  verbose=3)
            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #extracting the best parameters
            self.kernel = self.grid.best_params_['kernel']
            self.gamma = self.grid.best_params_['gamma']
            self.C = self.grid.best_params_['C']
            self.degree = self.grid.best_params_['degree']

            #creating a new model with the best parameters
            self.svm = SVC(kernel=self.kernel, gamma=self.gamma ,C=self.C,
                                             degree=self.degree)
            # training the mew model
            self.svm.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'SVM best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_svm method of the Model_Finder class')

            return self.svm
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_svm method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Support Vector Classifier Parameter tuning  failed. Exited the get_best_params_for_svm method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_naive_bayes(self,train_x,train_y):
        """
                                Method Name: get_best_params_for_naive_bayes
                                Description: get the parameters for Gaussian Naive Bayes Algorithm which give the best accuracy.
                                             Use Hyper Parameter Tuning.
                                Output: The model with the best parameters
                                On Failure: Raise Exception

                                Written By: Sanjeevan Thorat
                                Version: 1.0
                                Revisions: None

                        """
        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_naive_bayes method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {'var_smoothing':[1e-10,1e-9,1e-8,1e-7,1e-6,1e-5]}

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.nb, param_grid=self.param_grid, cv=5,  verbose=3)
            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #extracting the best parameters
            self.var_smoothing=self.grid.best_params_['var_smoothing']

            #creating a new model with the best parameters
            self.nb = GaussianNB(var_smoothing=self.var_smoothing)
            # training the mew model
            self.nb.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'Naive Bayes best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_naive_bayes method of the Model_Finder class')

            return self.nb
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_naive_bayes method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Naive Bayes Parameter tuning  failed. Exited the get_best_params_for_naive_bayes method of the Model_Finder class')
            raise Exception()
    def get_best_params_for_catboost(self,train_x,train_y):
        """
                                Method Name: get_best_params_for_catboost
                                Description: get the parameters for catboost Algorithm which give the best accuracy.
                                             Use Hyper Parameter Tuning.
                                Output: The model with the best parameters
                                On Failure: Raise Exception

                                Written By: Sanjeevan Thorat
                                Version: 1.0
                                Revisions: None

                        """
        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_catboost method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {'max_depth':[2,4,5,6],
                               'n_estimators':[10,30,50,125,200],
                               'learning_rate': [0.5, 0.1, 0.01, 0.001]
            }

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.catboost, param_grid=self.param_grid, cv=5,  verbose=3)
            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #extracting the best parameters

            self.learning_rate = self.grid.best_params_['learning_rate']
            self.max_depth = self.grid.best_params_['max_depth']
            self.n_estimators = self.grid.best_params_['n_estimators']

            #creating a new model with the best parameters
            self.catboost = CatBoostClassifier(learning_rate=self.learning_rate, max_depth=self.max_depth, n_estimators=self.n_estimators)
            # training the mew model
            self.catboost.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'Random Forest best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_random_forest method of the Model_Finder class')

            return self.catboost
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_catboost method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Catboost Parameter tuning  failed. Exited the get_best_params_for_catboost method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_random_forest(self,train_x,train_y):
        """
                                Method Name: get_best_params_for_random_forest
                                Description: get the parameters for Random Forest Algorithm which give the best accuracy.
                                             Use Hyper Parameter Tuning.
                                Output: The model with the best parameters
                                On Failure: Raise Exception

                                Written By: Sanjeevan Thorat
                                Version: 1.0
                                Revisions: None

                        """
        self.logger_object.log(self.file_object, 'Entered the get_best_params_for_random_forest method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid = {"n_estimators": [10, 50, 100, 130], "criterion": ['gini', 'entropy'],
                               "max_depth": range(2, 4, 1), "max_features": ['auto', 'log2']}

            #Creating an object of the Grid Search class
            self.grid = GridSearchCV(estimator=self.clf, param_grid=self.param_grid, cv=5,  verbose=3)
            #finding the best parameters
            self.grid.fit(train_x, train_y)

            #extracting the best parameters
            self.criterion = self.grid.best_params_['criterion']
            self.max_depth = self.grid.best_params_['max_depth']
            self.max_features = self.grid.best_params_['max_features']
            self.n_estimators = self.grid.best_params_['n_estimators']

            #creating a new model with the best parameters
            self.clf = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion,
                                              max_depth=self.max_depth, max_features=self.max_features)
            # training the mew model
            self.clf.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'Random Forest best params: '+str(self.grid.best_params_)+'. Exited the get_best_params_for_random_forest method of the Model_Finder class')

            return self.clf
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_random_forest method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Random Forest Parameter tuning  failed. Exited the get_best_params_for_random_forest method of the Model_Finder class')
            raise Exception()

    def get_best_params_for_xgboost(self,train_x,train_y):

        """
                                        Method Name: get_best_params_for_xgboost
                                        Description: get the parameters for XGBoost Algorithm which give the best accuracy.
                                                     Use Hyper Parameter Tuning.
                                        Output: The model with the best parameters
                                        On Failure: Raise Exception

                                        Written By: Sanjeevan Thorat
                                        Version: 1.0
                                        Revisions: None

                                """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_params_for_xgboost method of the Model_Finder class')
        try:
            # initializing with different combination of parameters
            self.param_grid_xgboost = {

                'learning_rate': [0.5, 0.1, 0.01, 0.001],
                'max_depth': [3, 5, 10, 20],
                'n_estimators': [10, 50, 100, 200]

            }
            # Creating an object of the Grid Search class
            self.grid= GridSearchCV(XGBClassifier(objective='binary:logistic'),self.param_grid_xgboost, verbose=3,cv=5)
            # finding the best parameters
            self.grid.fit(train_x, train_y)

            # extracting the best parameters
            self.learning_rate = self.grid.best_params_['learning_rate']
            self.max_depth = self.grid.best_params_['max_depth']
            self.n_estimators = self.grid.best_params_['n_estimators']

            # creating a new model with the best parameters
            self.xgb = XGBClassifier(learning_rate=self.learning_rate, max_depth=self.max_depth, n_estimators=self.n_estimators)
            # training the mew model
            self.xgb.fit(train_x, train_y)
            self.logger_object.log(self.file_object,
                                   'XGBoost best params: ' + str(
                                       self.grid.best_params_) + '. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            return self.xgb
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_xgboost method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'XGBoost Parameter tuning  failed. Exited the get_best_params_for_xgboost method of the Model_Finder class')
            raise Exception()


    def get_best_model(self,train_x,train_y,test_x,test_y):
        """
                                                Method Name: get_best_model
                                                Description: Find out the Model which has the best AUC score.
                                                Output: The best model name and the model object
                                                On Failure: Raise Exception

                                                Written By: Sanjeevan Thorat
                                                Version: 1.0
                                                Revisions: None

                                        """
        self.logger_object.log(self.file_object,
                               'Entered the get_best_model method of the Model_Finder class')

        try:
            # create best model for SVM
            self.svm = self.get_best_params_for_svm(train_x, train_y)
            self.prediction_svm = self.svm.predict(test_x)  # Predictions using the SVM Model

            if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.svm_score = accuracy_score(test_y, self.prediction_svm)
                self.logger_object.log(self.file_object, 'Accuracy for SVM:' + str(self.svm_score))  # Log AUC
            else:
                self.svm_score = roc_auc_score(test_y, self.prediction_svm)  # AUC for svm
                self.logger_object.log(self.file_object, 'AUC for SVM:' + str(self.svm_score))  # Log AUC

            # create best model for Logistic Regression
            self.logistic = self.get_best_params_for_logistic_regression(train_x, train_y)
            self.prediction_logistic = self.logistic.predict(test_x)  # Predictions using the logistic Model

            if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.logistic_score = accuracy_score(test_y, self.prediction_logistic)
                self.logger_object.log(self.file_object,
                                       'Accuracy for logistic:' + str(self.logistic_score))  # Log AUC
            else:
                self.logistic_score = roc_auc_score(test_y, self.prediction_logistic)  # AUC for logistic regression
                self.logger_object.log(self.file_object, 'AUC for logistic:' + str(self.logistic_score))  # Log AUC

            # create best model for Naive Bayes
            self.nb = self.get_best_params_for_naive_bayes(train_x, train_y)
            self.prediction_nb = self.nb.predict(test_x)  # Predictions using the XGBoost Model

            if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.nb_score = accuracy_score(test_y, self.prediction_nb)
                self.logger_object.log(self.file_object,
                                       'Accuracy for Naive Bayes:' + str(self.nb_score))  # Log AUC
            else:
                self.nb_score = roc_auc_score(test_y, self.prediction_nb)  # AUC for Naive Bayes
                self.logger_object.log(self.file_object, 'AUC for Naive Bayes:' + str(self.nb_score))  # Log AUC

            # create best model for Catboost
            self.catboost = self.get_best_params_for_catboost(train_x, train_y)
            self.prediction_catboost = self.catboost.predict(test_x)  # Predictions using the CatBoost Model

            if len(test_y.unique()) == 1:  # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.catboost_score = accuracy_score(test_y, self.prediction_catboost)
                self.logger_object.log(self.file_object,
                                       'Accuracy for CatBoost:' + str(self.catboost_score))  # Log AUC
            else:
                self.catboost_score = roc_auc_score(test_y, self.prediction_catboost)  # AUC for CatBoost
                self.logger_object.log(self.file_object, 'AUC for CatBoost:' + str(self.catboost_score))  # Log AUC

            # create best model for XGBoost
            self.xgboost= self.get_best_params_for_xgboost(train_x,train_y)
            self.prediction_xgboost = self.xgboost.predict(test_x) # Predictions using the XGBoost Model

            if len(test_y.unique()) == 1: #if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.xgboost_score = accuracy_score(test_y, self.prediction_xgboost)
                self.logger_object.log(self.file_object, 'Accuracy for XGBoost:' + str(self.xgboost_score))  # Log AUC
            else:
                self.xgboost_score = roc_auc_score(test_y, self.prediction_xgboost) # AUC for XGBoost
                self.logger_object.log(self.file_object, 'AUC for XGBoost:' + str(self.xgboost_score)) # Log AUC

            # create best model for Random Forest
            self.random_forest=self.get_best_params_for_random_forest(train_x,train_y)
            self.prediction_random_forest=self.random_forest.predict(test_x) # prediction using the Random Forest Algorithm

            if len(test_y.unique()) == 1:#if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
                self.random_forest_score = accuracy_score(test_y,self.prediction_random_forest)
                self.logger_object.log(self.file_object, 'Accuracy for RF:' + str(self.random_forest_score))
            else:
                self.random_forest_score = roc_auc_score(test_y, self.prediction_random_forest) # AUC for Random Forest
                self.logger_object.log(self.file_object, 'AUC for RF:' + str(self.random_forest_score))

            #comparing the two models
            model_list=[self.svm_score,self.logistic_score,self.nb_score,self.catboost_score,self.xgboost_score,self.random_forest_score]
            max_value= max(model_list)

            max_index = model_list.index(max_value)
            if max_index==0:
                return 'SVM',self.svm_score
            elif max_index==1:
                return 'LogisticRegression',self.logistic_score
            elif max_index==2:
                return 'Naive Bayes',self.nb_score
            elif max_index==3:
                return 'CatBoost',self.catboost_score
            elif max_index==4:
                return 'XgBoost',self.xgboost_score
            elif max_index==5:
                return 'Random Forest',self.random_forest_score

        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()

