

from django import forms
from .models import Enquiry

class SentimentAnalysisForm(forms.Form):
    url = forms.URLField(required=False, label='URL')
    file = forms.FileField(required=False, label='File')

    def clean(self):
        cleaned_data = super().clean()
        url = cleaned_data.get('url')
        file = cleaned_data.get('file')

        if not url and not file:
            raise forms.ValidationError('Please provide either a URL or a file.')

        return cleaned_data
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for field_name in self.fields:
            self.fields[field_name].widget.attrs.update({'class': 'form-control'})




class ContactForm(forms.ModelForm):
    class Meta:
        model = Enquiry
        fields = ['name', 'email', 'phone', 'service', 'message']

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        placeholders = {
            'name': 'Enter your name',
            'email': 'Enter email address',
            'phone': 'Enter phone number',
            'service': 'Enter Service',
            'message': 'Enter message',
        }
        for field_name in self.fields:
            self.fields[field_name].widget.attrs.update({
                'class': 'form-control',
                'placeholder': placeholders.get(field_name, ''),
            })

