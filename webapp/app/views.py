from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from . import util
# Create your views here.


def home(request):
    return render(request,'app.html')

@csrf_exempt
def classify(request):
    if request.method == "POST":
        arr = util.classify_image(request.POST['image_data'])
        if len(arr)==0:
            arr = {'bool':0}
        print(arr)
        return JsonResponse(arr)
        
    
