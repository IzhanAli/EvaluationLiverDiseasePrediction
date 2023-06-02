from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, HttpResponse
from django.contrib import messages
from sklearn.tree import DecisionTreeClassifier

from .forms import UserRegistrationForm
from .models import UserRegistrationModel


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHomePage.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHomePage.html', {})


def PreProcess(request):
    import matplotlib.pyplot as plt
    import seaborn as sns
    # inline
    import pandas as pd
    from django.conf import settings
    import os
    path = os.path.join(settings.MEDIA_ROOT, 'indian_liver_patient.csv')
    data = pd.read_csv(path)
    # checking the stats
    # given in the website 416 liver disease patients and 167 non liver disease patients
    # need to remap the classes liver disease:=1 and no liver disease:=0 (normal convention to be followed)
    data['Dataset'] = data['Dataset'].map(
        {2: 0, 1: 1})
    count_classes = pd.value_counts(data['Dataset'], sort=True).sort_index()
    count_classes.plot(kind='bar')
    # plt.title("Liver disease classes bar graph")
    # plt.xlabel("Dataset")
    # plt.ylabel("Frequency")
    # plt.savefig('classlabels.png')
    # data['Dataset'] = data['Dataset'].map({2: 0, 1: 1})
    data['Albumin_and_Globulin_Ratio'].fillna(value=0, inplace=True)
    data_features = data.drop(['Dataset'], axis=1)
    data_num_features = data.drop(['Gender', 'Dataset'], axis=1)
    data_num_features.head()
    data_num_features.describe()  # check to whether feature scaling has to be performed or not
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    cols = list(data_num_features.columns)
    data_features_scaled = pd.DataFrame(data=data_features)
    data_features_scaled[cols] = scaler.fit_transform(data_features[cols])
    data_features_scaled.head()
    data_exp = pd.get_dummies(data_features_scaled)
    data_exp.head()
    # Set up the matplotlib figure
    # f, ax = plt.subplots(figsize=(12, 10))
    # plt.title('Pearson Correlation of liver disease Features')
    # # Draw the heatmap using seaborn
    # sns.heatmap(data_num_features.astype(float).corr(), linewidths=0.25, vmax=1.0, square=True, cmap="YlGnBu",
    #             linecolor='black', annot=True)
    # plt.savefig('corr.png')
    # plt.show()
    return render(request, 'PreProcess.html', {"data": data_num_features.to_html})


def View_Data(request):
    import pandas as pd
    from django.conf import settings
    import os
    path = os.path.join(settings.MEDIA_ROOT, 'indian_liver_patient.csv')
    df = pd.read_csv(path)
    df = df[['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
             'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio']]
    df = df.head(584).to_html(classes='table-sm')
    return render(request, 'users/readDataset.html', {'data': df})


def ML(request):
    return render(request, 'users/ML.html')


def MLResult(request):
    if request.method == "POST":
        from django.conf import settings
        import pandas as pd
        age = request.POST.get('age')
        gender = request.POST.get('gender')
        Total_Bilirubin = request.POST.get('Total_Bilirubin')
        Direct_Bilirubin = request.POST.get('Direct_Bilirubin')
        Alkaline_Phosphotase = request.POST.get('Alkaline_Phosphotase')
        Alamine_Aminotransferase = request.POST.get('Alamine_Aminotransferase')
        Aspartate_Aminotransferase = request.POST.get('Aspartate_Aminotransferase')
        Total_Protiens = request.POST.get('Total_Protiens')
        Albumin = request.POST.get('Albumin')
        Albumin_and_Globulin_Ratio = request.POST.get('Albumin_and_Globulin_Ratio')

        path = settings.MEDIA_ROOT + "\\" + "indian_liver_patient.csv"
        data = pd.read_csv(path, delimiter=',')
        x = data.iloc[:, 0:10]
        y = data.iloc[:, -1]
        x = pd.get_dummies(x)

        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        x_train = sc.fit_transform(x_train)
        x_test = sc.fit_transform(x_test)
        x_train = pd.DataFrame(x_train)
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
        test_set = [age, Total_Bilirubin, Direct_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase,
                    Aspartate_Aminotransferase, Total_Protiens, Albumin, Albumin_and_Globulin_Ratio, gender, gender]
        x_train = x_train.fillna(x_train.mean())
        model.fit(x_train, y_train)
        test_set = pd.Series(test_set).fillna(0).tolist()
        y_pred = model.predict([test_set])
        if y_pred == 1:
            msg = "HI,  YOUR   HEALTH  IS  ABSOLUTELY  FINE"
        elif y_pred == 2:
            msg = "HELLO,  YOUR  HEALTH  IS   NOT   SUFFICIENT"
        return render(request, 'users/ML.html', {'msg': msg})


def user_classification_results(request):
    from .utility import ProcessLiverDiseas
    lg_accuracy, lg_precision,lg_sensitivity, lg_specificity = ProcessLiverDiseas.start_logistic_regression()
    svm_accuracy, svm_precision, svm_sensitivity, svm_specificity = ProcessLiverDiseas.start_svm()
    lg = {"lg_accuracy": lg_accuracy, "lg_precision": lg_precision, "lg_sensitivity": lg_sensitivity, "lg_specificity": lg_specificity}
    svm = {"svm_accuracy": svm_accuracy, "svm_precision": svm_precision, "svm_sensitivity": svm_sensitivity, "svm_specificity": svm_specificity}
    return render(request, "users/classification_report.html",{"lg": lg, "svm": svm})
