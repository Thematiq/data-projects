from django.http import HttpResponse, Http404
from django.views.decorators.csrf import csrf_exempt
from django.template import loader
from django.urls import reverse
from django.shortcuts import redirect
from urllib.parse import urlencode

from .models import static_engine


@csrf_exempt
def index(request):
    if request.method == 'POST':
        print(request.POST)
        base = reverse('search')
        query = urlencode({
            'query': request.POST.get('text'),
            'size': request.POST.get('size'),
            'type': request.POST.get('radio'),
            'svd': request.POST.get('svd', 0),
            'rank': request.POST.get('rank')
        })
        url = '{}?{}'.format(base, query)
        return redirect(url)
    context = {
        'rank': static_engine.max_k
    }
    template = loader.get_template('index.html')
    return HttpResponse(template.render(context, request))


def search(request):
    try:
        query = request.GET.get('query')
        size = int(request.GET.get('size', 5))
        model = int(request.GET.get('type', 0)) * 2 + int(request.GET.get('svd', 0))
        k = int(request.GET.get('rank'))
    except Exception as e:
        print(e)
        return redirect(reverse('error'))

    pages = {
        # Count + no SVD
        0: static_engine.normal_query,
        # Count + SVD
        1: lambda *x: static_engine.svd_query(*x, k, False),
        # IDF + no SVD
        2: static_engine.idf_query,
        # IDF + SVD
        3: lambda *x: static_engine.svd_query(*x, k, True)
    }[model](query, size)

    context = {
        'search_results': pages,
        'query': query
    }
    template = loader.get_template('search.html')
    return HttpResponse(template.render(context, request))


def error(request):
    template = loader.get_template('error.html')
    return HttpResponse(template.render({}, request))
