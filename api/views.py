from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from rest_framework.generics import CreateAPIView
from rest_framework.parsers import MultiPartParser, FormParser
from .serializers import InferenceRequestSerializer
from .core.inference import predict

# Create your views here.
class InferenceCreateAPIView(CreateAPIView):
    
    serializer_class = InferenceRequestSerializer
    parser_classes = (MultiPartParser, FormParser)
    
    def create(self, request, *args, **kwargs):
        
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        file = request.FILES['image']
        results = predict(file)
        return Response(data=results, status=status.HTTP_200_OK)
    


