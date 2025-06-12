import json

# Load scalability results
with open('scalability_results/scalability_results.json', 'r') as f:
    scalability_data = json.load(f)

# Extract metrics in the expected format
extracted_data = []
for exp in scalability_data:
    if exp.get('status') == 'success' and 'attack_results' in exp:
        # Extract attack success metrics from each attack result
        success_metrics = []
        for attack in exp['attack_results']:
            if 'attack_success_metric' in attack:
                success_metrics.append(attack['attack_success_metric'])
        
        if success_metrics:
            extracted_exp = {
                'config': exp['config'],
                'success_metrics': success_metrics,
                'experiment_id': exp.get('experiment_id'),
                'status': exp['status']
            }
            extracted_data.append(extracted_exp)

# Save extracted metrics
with open('scalability_results/extracted_metrics.json', 'w') as f:
    json.dump(extracted_data, f, indent=2)

print(f"Extracted metrics from {len(extracted_data)} experiments")
print(f"Each experiment has {len(extracted_data[0]['success_metrics']) if extracted_data else 0} attack results")