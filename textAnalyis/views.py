from django.shortcuts import redirect,render
from webApp.forms import ContactForm
from django.contrib import messages


def index(request):
    return render(request, "index.html")


def contact(request):
    if request.method == 'POST':
        form = ContactForm(request.POST)
        if form.is_valid():
            form.save()
            # Add a success message
            messages.success(request, 'Form submitted successfully!')
            return redirect('contact')  
    else:
        form = ContactForm()
    return render(request, "contact.html", {'form': form})
