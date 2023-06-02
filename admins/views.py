from django.shortcuts import render
from django.shortcuts import render
from django.contrib import messages
from users.models import UserRegistrationModel
# Create your views here.
def AdminLoginCheck(request):
    if request.method == 'POST':
        usrid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("User ID is = ", usrid)
        if usrid == 'admin' and pswd == 'admin':
            return render(request, 'admins/AdminHome.html')

        else:
            messages.success(request, 'Please Check Your Login Details')
    return render(request, 'AdminLogin.html', {})

def AdminHome(request):
    return render(request, 'admins/AdminHome.html')

def RegisterUsersView(request):
    data = UserRegistrationModel.objects.all()
    return render(request,'admins/viewregisterusers.html',{'data':data})


def ActivaUsers(request):
    if request.method == 'GET':
        id = request.GET.get('uid')
        status = 'activated'
        print("PID = ", id, status)
        UserRegistrationModel.objects.filter(id=id).update(status=status)
        data = UserRegistrationModel.objects.all()
        return render(request,'admins/viewregisterusers.html',{'data':data})


def admin_view_results(request):
    from users.utility import ProcessLiverDiseas
    lg_accuracy, lg_precision, lg_sensitivity, lg_specificity = ProcessLiverDiseas.start_logistic_regression()
    svm_accuracy, svm_precision, svm_sensitivity, svm_specificity = ProcessLiverDiseas.start_svm()
    lg = {"lg_accuracy": lg_accuracy, "lg_precision": lg_precision, "lg_sensitivity": lg_sensitivity,
          "lg_specificity": lg_specificity}
    svm = {"svm_accuracy": svm_accuracy, "svm_precision": svm_precision, "svm_sensitivity": svm_sensitivity,
           "svm_specificity": svm_specificity}
    return render(request, "admins/reports.html", {"lg": lg, "svm": svm})
