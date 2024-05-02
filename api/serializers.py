from rest_framework import serializers

class InferenceRequestSerializer(serializers.Serializer):
    
    image = serializers.ImageField()
        