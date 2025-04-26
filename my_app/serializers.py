from rest_framework import serializers

class PredictionSerializer(serializers.Serializer):
    Num_Children = serializers.ListField(child=serializers.IntegerField())
    Gender = serializers.ListField(child=serializers.ChoiceField(choices=["Male", "Female"]))
    Income = serializers.ListField(child=serializers.IntegerField())
    Own_Car = serializers.ListField(child=serializers.ChoiceField(choices=["Yes", "No"]))
    Own_Housing = serializers.ListField(child=serializers.ChoiceField(choices=["Yes", "No"]))
