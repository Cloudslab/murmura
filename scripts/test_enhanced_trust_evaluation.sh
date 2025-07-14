#!/bin/bash

# Enhanced Trust Monitoring Evaluation Script
# Tests trust monitoring with various node counts and includes baseline comparisons
# For EdgeDrift paper evaluation

set -e  # Exit on any error

echo "=========================================="
echo "Enhanced Trust Monitoring Evaluation"
echo "=========================================="
echo "Testing: MNIST, CIFAR-10"
echo "Node counts: 10, 20, 30, 50"
echo "Topologies: ring, complete, line"
echo "Attacks: Gradient manipulation + Label flipping (30% malicious)"
echo "Baselines: With and without trust monitoring"
echo "=========================================="

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="enhanced_trust_results_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR
cd $OUTPUT_DIR

# Node count configurations
NODE_COUNTS=(10 20 30 50)

# Common parameters for all experiments  
COMMON_BASE_PARAMS="
    --malicious_clients_ratio 0.3
    --attack_intensity_start 0.2
    --attack_intensity_end 1.0
    --intensity_progression linear
    --attack_start_round 2
    --rounds 10
    --epochs 1
    --create_animation
    --malicious_node_seed 42
    --data_partitioning_seed 42
"

# Trust monitoring parameters (using new state-based weighting with less aggressive decay)
TRUST_PARAMS="
    --enable_trust_monitoring
    --enable_trust_weighted_aggregation
    --enable_exponential_decay
    --exponential_decay_base 0.75
"

# Gradient manipulation specific parameters
GRADIENT_PARAMS="
    --attack_type gradient_manipulation
    --gradient_noise_scale 1.0
    --gradient_sign_flip_prob 0.1
"

# Label flipping specific parameters
LABEL_FLIP_PARAMS="
    --attack_type label_flipping
    --label_flip_target 1
"

# Function to extract trust score evolution from experiment logs
extract_trust_evolution() {
    local log_file=$1
    local base_name=$(basename "$log_file" .txt)
    local trust_file="${base_name}_trust_evolution.txt"
    
    echo "   📈 Extracting trust score evolution to: $trust_file"
    
    # Use Python for more reliable parsing
    python3 << EOF
import re
import sys

log_file = "$log_file"
trust_file = "$trust_file"

# Write header
with open(trust_file, 'w') as f:
    f.write("# Trust Score Evolution for $base_name\n")
    f.write("# Format: Round,Node,Neighbor,Trust_Score\n")
    f.write("# Generated at: $(date)\n")
    f.write("\n")

current_round = None
trust_entries = []

try:
    with open(log_file, 'r') as f:
        for line in f:
            # Check for round markers
            round_match = re.search(r'Round (\d+): Processing trust monitoring', line)
            if round_match:
                current_round = round_match.group(1)
                continue
            
            # Check for trust score lines
            trust_match = re.search(r'Node (\d+): Trust scores: ({.*})', line)
            if trust_match and current_round:
                node_num = trust_match.group(1)
                trust_dict_str = trust_match.group(2)
                
                # Parse the trust dictionary
                try:
                    # Clean up the string and evaluate as dict
                    trust_dict_str = trust_dict_str.replace("'", '"')
                    import json
                    trust_dict = json.loads(trust_dict_str)
                    
                    # Extract trust scores
                    for neighbor_node, trust_score in trust_dict.items():
                        # Remove 'node_' prefix if present
                        neighbor_id = neighbor_node.replace('node_', '')
                        trust_entries.append((int(current_round), int(node_num), int(neighbor_id), float(trust_score)))
                except:
                    # Fallback parsing for malformed JSON
                    import ast
                    try:
                        trust_dict = ast.literal_eval(trust_dict_str)
                        for neighbor_node, trust_score in trust_dict.items():
                            neighbor_id = neighbor_node.replace('node_', '')
                            trust_entries.append((int(current_round), int(node_num), int(neighbor_id), float(trust_score)))
                    except:
                        continue

    # Remove duplicates and sort entries by round, node, neighbor
    trust_entries = list(set(trust_entries))
    trust_entries.sort(key=lambda x: (x[0], x[1], x[2]))
    
    # Write entries to file
    with open(trust_file, 'a') as f:
        for round_num, node, neighbor, score in trust_entries:
            f.write(f"{round_num},{node},{neighbor},{score}\n")
    
    print(f"   📊 Extracted {len(trust_entries)} trust score entries")
    
    if trust_entries:
        print("   📝 Sample entries:")
        for i, (round_num, node, neighbor, score) in enumerate(trust_entries[:4]):
            print(f"      {round_num},{node},{neighbor},{score}")
    
except Exception as e:
    print(f"   ⚠️  Error extracting trust scores: {e}")
    
EOF
    
    # Check if extraction was successful
    if [ ! -f "$trust_file" ] || [ ! -s "$trust_file" ]; then
        echo "   ⚠️  No trust scores found in log file"
        rm -f "$trust_file"
    fi
}

# Function to run experiment
run_experiment() {
    local dataset=$1
    local topology=$2
    local attack_type=$3
    local node_count=$4
    local trust_enabled=$5
    
    local trust_suffix=""
    if [ "$trust_enabled" = "true" ]; then
        trust_suffix="_trust"
    else
        trust_suffix="_baseline"
    fi
    
    local output_file="${dataset}_${topology}_${attack_type}_n${node_count}${trust_suffix}.txt"
    
    echo ""
    echo "===========================================" 
    echo "Running: $dataset with $topology topology ($attack_type attack)"
    echo "Nodes: $node_count, Trust: $trust_enabled"
    echo "Output: $output_file"
    echo "Started: $(date)"
    echo "==========================================="
    
    # Select attack parameters
    if [ "$attack_type" = "gradient" ]; then
        ATTACK_PARAMS="$GRADIENT_PARAMS"
    elif [ "$attack_type" = "label_flip" ]; then
        ATTACK_PARAMS="$LABEL_FLIP_PARAMS"
    fi
    
    # Select trust parameters
    if [ "$trust_enabled" = "true" ]; then
        EXPERIMENT_PARAMS="$COMMON_BASE_PARAMS $TRUST_PARAMS $ATTACK_PARAMS --num_actors $node_count"
    else
        EXPERIMENT_PARAMS="$COMMON_BASE_PARAMS $ATTACK_PARAMS --num_actors $node_count"
    fi
    
    # Run the experiment (without timeout for macOS compatibility)
    if [ "$dataset" = "mnist" ]; then
        python ../../murmura/examples/dp_decentralized_mnist_example.py \
            --topology $topology \
            $EXPERIMENT_PARAMS \
            > $output_file 2>&1
    elif [ "$dataset" = "cifar10" ]; then
        python ../../murmura/examples/dp_decentralized_cifar10_example.py \
            --topology $topology \
            $EXPERIMENT_PARAMS \
            > $output_file 2>&1
    fi
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✅ SUCCESS: $dataset with $topology topology (n=$node_count, trust=$trust_enabled)"
        
        # Extract key metrics
        echo "📊 Results Summary:"
        
        # Extract actual malicious node indices
        local malicious_indices=$(grep "Malicious clients will be created at indices:" $output_file | grep -o "\[.*\]" || echo "Not found")
        echo "   🎯 Actual malicious nodes: $malicious_indices"
        
        if [ "$trust_enabled" = "true" ]; then
            # Extract all unique detected suspicious neighbors across all rounds
            local all_detections=$(grep -o "detected suspicious neighbors: \['node_[0-9]*'\]" $output_file | grep -o "node_[0-9]*" | sed 's/node_//' | sort -n | uniq)
            if [ -n "$all_detections" ]; then
                local detected_array="[$(echo $all_detections | tr ' ' ',')]"
                echo "   🔍 Detected suspicious nodes: $detected_array"
                
                # Count total detections
                local detection_count=$(echo $all_detections | wc -w | tr -d ' ')
                echo "   📈 Total unique detections: $detection_count"
            else
                echo "   🔍 Detected suspicious nodes: None"
                echo "   📈 Total unique detections: 0"
            fi
            
            # First detection round
            local first_detection=$(grep -B 50 "detected suspicious neighbors" $output_file | grep "Round [0-9]" | head -1 | grep -o "Round [0-9]*" || echo "Not detected")
            echo "   ⏱️  First detection: $first_detection"
        else
            echo "   🔍 Trust monitoring: DISABLED (baseline)"
        fi
        
        echo "   🎯 Final accuracy: $(grep "Final Test Accuracy.*Honest Nodes" $output_file | tail -1 || grep "Final Test Accuracy" $output_file | tail -1 || echo "Not found")"
        echo "   📊 Accuracy improvement: $(grep "Accuracy Improvement.*Honest Nodes" $output_file | tail -1 || grep "Accuracy Improvement" $output_file | tail -1 || echo "Not found")"
        
        # Extract trust score evolution for trust-enabled experiments
        if [ "$trust_enabled" = "true" ]; then
            extract_trust_evolution $output_file
        fi
        
        echo "   ✅ Completed: $(date)"
    elif [ $exit_code -eq 124 ]; then
        echo "⏰ TIMEOUT: $dataset with $topology topology (n=$node_count, trust=$trust_enabled)"
        echo "   Experiment exceeded 30 minutes time limit"
    else
        echo "❌ FAILED: $dataset with $topology topology (n=$node_count, trust=$trust_enabled)"
        echo "   Check $output_file for error details"
    fi
    echo ""
}

# Function to run scalability experiments
run_scalability_experiments() {
    echo ""
    echo "🚀 SCALABILITY EXPERIMENTS"
    echo "=========================="
    
    datasets=("mnist" "cifar10")
    topologies=("ring" "complete")  # Skip line topology for scalability (too slow)
    attack_types=("gradient" "label_flip")
    trust_settings=("true" "false")
    
    for dataset in "${datasets[@]}"; do
        for topology in "${topologies[@]}"; do
            for attack_type in "${attack_types[@]}"; do
                for node_count in "${NODE_COUNTS[@]}"; do
                    for trust_enabled in "${trust_settings[@]}"; do
                        run_experiment $dataset $topology $attack_type $node_count $trust_enabled
                    done
                done
            done
        done
    done
}

# Function to run baseline comparison experiments 
run_baseline_experiments() {
    echo ""
    echo "📊 BASELINE COMPARISON EXPERIMENTS"
    echo "=================================="
    echo "Running focused experiments to compare trust vs baseline performance"
    
    # Focused set for detailed comparison
    datasets=("mnist" "cifar10")
    topologies=("complete")  # Complete topology for best trust performance
    attack_types=("gradient" "label_flip")
    node_counts=(10 30)  # Small and medium scale
    trust_settings=("true" "false")
    
    for dataset in "${datasets[@]}"; do
        for topology in "${topologies[@]}"; do
            for attack_type in "${attack_types[@]}"; do
                for node_count in "${node_counts[@]}"; do
                    for trust_enabled in "${trust_settings[@]}"; do
                        run_experiment $dataset $topology $attack_type $node_count $trust_enabled
                    done
                done
            done
        done
    done
}

# Main execution
echo "Starting experiments at: $(date)"
echo "Output directory: $(pwd)"
echo ""

# Run scalability experiments
run_scalability_experiments

# Run focused baseline experiments
run_baseline_experiments

echo "=========================================="
echo "All experiments completed at: $(date)"
echo "=========================================="

# Generate comprehensive summary report
echo ""
echo "📋 COMPREHENSIVE SUMMARY REPORT"
echo "==============================="

# Create CSV summary file
csv_file="experiment_summary.csv"
echo "Dataset,Topology,Attack,Nodes,Trust,Actual_Malicious,Detected_Malicious,Detection_Count,Detection_Round,Final_Accuracy,Accuracy_Improvement,Status" > $csv_file

for file in *.txt; do
    if [ -f "$file" ]; then
        # Parse filename to extract experiment parameters
        filename=$(basename "$file" .txt)
        
        # Extract parameters using regex-like approach
        dataset=$(echo $filename | cut -d'_' -f1)
        topology=$(echo $filename | cut -d'_' -f2)
        attack=$(echo $filename | cut -d'_' -f3)
        node_info=$(echo $filename | cut -d'_' -f4)
        trust_info=$(echo $filename | cut -d'_' -f5)
        
        # Extract node count
        node_count=$(echo $node_info | sed 's/n//')
        
        # Determine trust status
        if [[ "$trust_info" == "trust" ]]; then
            trust_enabled="true"
        else
            trust_enabled="false"
        fi
        
        echo ""
        echo "🔍 $dataset ($topology, $attack, n=$node_count, trust=$trust_enabled):"
        
        # Check if experiment succeeded
        if grep -q "SUCCESS" $file || grep -q "Final Test Accuracy" $file; then
            status="SUCCESS"
            
            # Extract actual malicious node indices (same for baseline and trust experiments)
            actual_malicious=$(grep "Malicious clients will be created at indices:" $file | grep -o "\[.*\]" | tr -d '[]' | tr -d ' ' || echo "N/A")
            
            # Detection results
            if [ "$trust_enabled" = "true" ]; then
                # Extract all unique detected suspicious neighbors
                detected_nodes=$(grep -o "detected suspicious neighbors: \['node_[0-9]*'\]" $file | grep -o "node_[0-9]*" | sed 's/node_//' | sort -n | uniq)
                if [ -n "$detected_nodes" ]; then
                    detected_malicious=$(echo "$detected_nodes" | tr '\n' ',' | sed 's/,$//')
                    detection_count=$(echo $detected_nodes | wc -w | tr -d ' ')
                else
                    detected_malicious="None"
                    detection_count="0"
                fi
                detection_round=$(grep -B 50 "detected suspicious neighbors" $file | grep "Round [0-9]" | head -1 | grep -o "[0-9]*" || echo "N/A")
            else
                detected_malicious="N/A"
                detection_count="N/A"
                detection_round="N/A"
            fi
            
            # Accuracy results (prefer honest nodes metrics, fallback to old format)
            final_acc=$(grep "Final Test Accuracy.*Honest Nodes" $file | tail -1 | grep -o "[0-9]*\.[0-9]*%" || grep "Final Test Accuracy" $file | tail -1 | grep -o "[0-9]*\.[0-9]*%" || echo "N/A")
            acc_improvement=$(grep "Accuracy Improvement.*Honest Nodes" $file | tail -1 | grep -o "[-]*[0-9]*\.[0-9]*%" || grep "Accuracy Improvement" $file | tail -1 | grep -o "[-]*[0-9]*\.[0-9]*%" || echo "N/A")
            
            echo "   🎯 Actual malicious: [$actual_malicious]"
            echo "   🔍 Detected malicious: [$detected_malicious]"
            echo "   📍 Detection: $detection_count malicious nodes in round $detection_round"
            echo "   🎯 Final Accuracy: $final_acc"
            echo "   📈 Accuracy Improvement: $acc_improvement"
            echo "   📄 File: $file"
            
            # Add to CSV
            echo "$dataset,$topology,$attack,$node_count,$trust_enabled,$actual_malicious,$detected_malicious,$detection_count,$detection_round,$final_acc,$acc_improvement,$status" >> $csv_file
        else
            status="FAILED"
            echo "   ❌ FAILED - Check $file for details"
            echo "$dataset,$topology,$attack,$node_count,$trust_enabled,N/A,N/A,N/A,N/A,N/A,N/A,$status" >> $csv_file
        fi
    fi
done

echo ""
echo "=========================================="
echo "📊 ANALYSIS AND NEXT STEPS"
echo "=========================================="
echo ""
echo "💾 All results saved in: $(pwd)"
echo "📈 Summary CSV file: $csv_file"
echo "🎬 Animation files (*.mp4) generated for successful experiments"
echo "📊 Trust evolution files (*_trust_evolution.txt) extracted for trust experiments"
echo ""
echo "🔍 Quick Analysis Commands:"
echo ""
echo "   # Compare trust vs baseline accuracy:"
echo "   grep 'Final Test Accuracy' *_trust.txt | sort"
echo "   grep 'Final Test Accuracy' *_baseline.txt | sort"
echo ""
echo "   # Verify seed consistency (same malicious nodes and data partitions):"
echo "   grep 'Malicious clients will be created' *_baseline.txt | cut -d: -f2 | sort | uniq"
echo "   grep 'Malicious clients will be created' *_trust.txt | cut -d: -f2 | sort | uniq"
echo "   # Both baseline and trust experiments use same seeds (malicious=42, data_partitioning=42)"
echo ""
echo "   # Detection performance across node counts:"
echo "   grep 'Trust monitoring detected.*suspicious neighbors' *.txt | sort"
echo ""
echo "   # Trust evolution analysis:"
echo "   echo \"Trust evolution files:\"; ls -la *_trust_evolution.txt 2>/dev/null || echo \"None found\""
echo ""
echo "   # Scalability analysis:"
echo "   for n in 10 20 30 50; do echo \"Node count \$n:\"; grep \"Final Test Accuracy\" *n\$n*.txt; done"
echo ""
echo "=========================================="