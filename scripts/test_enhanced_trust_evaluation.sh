#!/bin/bash

# Enhanced Trust Monitoring Evaluation Script
# Tests trust monitoring with various node counts and includes baseline comparisons
# For EdgeDrift paper evaluation

set -e  # Exit on any error

echo "=========================================="
echo "Enhanced Trust Monitoring Evaluation"
echo "=========================================="
echo "Testing: Eurosat, CIFAR-10"
echo "Node counts: 20, 30, 50"
echo "Topologies: ring, complete"
echo "Attacks: Gradient manipulation + Label flipping (30% malicious)"
echo "Training: 10 rounds, 2 epochs (faster evaluation)"
echo "Min partition: 100 samples (scaled for 50 nodes)"
echo "Resource monitoring: Enabled (CPU/Memory + Trust monitor overhead)"
echo "Comparisons: Trust-weighted vs baseline gossip averaging"
echo "=========================================="

# Create output directory with timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="enhanced_trust_results_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR
cd $OUTPUT_DIR

# Node count configurations
NODE_COUNTS=(10)

# Common parameters for all experiments  
COMMON_BASE_PARAMS="
    --malicious_clients_ratio 0.3
    --attack_intensity_start 0.2
    --attack_intensity_end 1.0
    --intensity_progression linear
    --attack_start_round 2
    --rounds 10
    --epochs 2
    --min_partition_size 100
    --monitor_resources
    --health_check_interval 10
    --create_animation
    --malicious_node 42
    --data_partitioning_seed 42
    --model_seed 42
"

# Trust monitoring parameters (using aggressive polynomial decay)
TRUST_PARAMS="
    --enable_trust_monitoring
    --enable_trust_weighted_aggregation
    --enable_trust_resource_monitoring
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
    
    # Select trust parameters and aggregation strategy
    if [ "$trust_enabled" = "true" ]; then
        EXPERIMENT_PARAMS="$COMMON_BASE_PARAMS $TRUST_PARAMS $ATTACK_PARAMS --num_actors $node_count --aggregation_strategy trust_weighted_gossip"
    else
        EXPERIMENT_PARAMS="$COMMON_BASE_PARAMS $ATTACK_PARAMS --num_actors $node_count --aggregation_strategy gossip_avg"
    fi
    
    # Run the experiment (without timeout for macOS compatibility)
    if [ "$dataset" = "eurosat" ]; then
        PYTHONUNBUFFERED=1 python ../../murmura/examples/dp_decentralized_eurosat_example.py \
            --topology $topology \
            $EXPERIMENT_PARAMS \
            --log_level INFO \
            > $output_file 2>&1
    elif [ "$dataset" = "cifar10" ]; then
        PYTHONUNBUFFERED=1 python ../../murmura/examples/dp_decentralized_cifar10_example.py \
            --topology $topology \
            $EXPERIMENT_PARAMS \
            --log_level INFO \
            > $output_file 2>&1
    fi
    
    local exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "✅ SUCCESS: $dataset with $topology topology (n=$node_count, trust=$trust_enabled)"
        
        # Extract key metrics
        echo "📊 Results Summary:"
        
        # Extract actual malicious node indices - improved parsing
        local malicious_indices=$(grep "Malicious clients will be created at indices:" $output_file | grep -o "\[.*\]" | head -1 || echo "Not found")
        echo "   🎯 Actual malicious nodes: $malicious_indices"
        
        if [ "$trust_enabled" = "true" ]; then
            # Extract detected suspicious neighbors - updated for new log format
            # Look for: "Trust monitoring detected X suspicious neighbors across all nodes: ['node_1', 'node_0']"
            local detected_line=$(grep "Trust monitoring detected.*suspicious neighbors across all nodes:" $output_file | tail -1)
            if [ -n "$detected_line" ]; then
                # Extract the node list from the line like: ['node_1', 'node_0', 'node_7']
                local all_detections=$(echo "$detected_line" | grep -o "\['[^']*'[^]]*\]" | sed "s/'node_/'/g" | sed "s/'//g" | sed 's/\[//' | sed 's/\]//' | tr ',' '\n' | sed 's/^ *//' | sort -n | uniq)
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
            else
                echo "   🔍 Detected suspicious nodes: None"
                echo "   📈 Total unique detections: 0"
            fi
            
            # First detection round - look for the global detection message
            local first_detection=$(grep -B 10 "Trust monitoring detected.*suspicious neighbors across all nodes:" $output_file | grep "Round [0-9]" | tail -1 | grep -o "[0-9]*" || echo "Not detected")
            echo "   ⏱️  First detection: Round $first_detection"
        else
            echo "   🔍 Trust monitoring: DISABLED (baseline)"
        fi
        
        # Fixed accuracy parsing - initial accuracy doesn't have "Honest Nodes" suffix
        local initial_acc=$(grep "Initial Test Accuracy:" $output_file | tail -1 | grep -o "[0-9]*\.[0-9]*%" || echo "Not found")
        local final_acc=$(grep "Final Test Accuracy.*Honest Nodes" $output_file | tail -1 | grep -o "[0-9]*\.[0-9]*%" || grep "Final Test Accuracy:" $output_file | tail -1 | grep -o "[0-9]*\.[0-9]*%" || echo "Not found")
        local acc_improvement=$(grep "Accuracy Improvement.*Honest Nodes" $output_file | tail -1 | grep -o "[-]*[0-9]*\.[0-9]*%" || grep "Accuracy Improvement:" $output_file | tail -1 | grep -o "[-]*[0-9]*\.[0-9]*%" || echo "Not found")
        
        echo "   🚀 Initial accuracy: $initial_acc"
        echo "   🎯 Final accuracy: $final_acc"
        echo "   📊 Accuracy improvement: $acc_improvement"
        
        # Extract aggregation strategy information
        echo "   🔄 Aggregation details:"
        local aggregation_strategy=$(grep "AGGREGATION_DEBUG.*Round 1:" $output_file | head -1 | grep -o "strategy=[A-Za-z]*" | sed 's/strategy=//' || echo "Unknown")
        local mixing_param=$(grep "AGGREGATION_DEBUG.*Round 1:" $output_file | head -1 | grep -o "mixing=[0-9.]*" | sed 's/mixing=//' || echo "Unknown")
        echo "      Strategy: $aggregation_strategy (mixing: $mixing_param)"
        
        # Count aggregation debug messages to verify consistency
        local trust_agg_count=$(grep -c "TRUST-WEIGHTED aggregation" $output_file || echo "0")
        local baseline_agg_count=$(grep -c "BASELINE aggregation" $output_file || echo "0")
        echo "      Trust-weighted calls: $trust_agg_count, Baseline calls: $baseline_agg_count"
        
        # Extract resource usage metrics
        echo "   💾 Resource usage:"
        echo "      $(grep "resource usage:" $output_file | tail -1 || echo "      Resource monitoring data not found")"
        
        # Extract trust monitor resource usage
        if [ "$trust_enabled" = "true" ]; then
            echo "   🧠 Trust monitor resources:"
            local trust_resource_found=false
            
            # Look for the new trust resource monitoring output
            if grep -q "Trust Monitor Resource Usage Summary" $output_file; then
                trust_resource_found=true
                echo "      Individual node summaries found in logs"
                
                # Check for aggregate summary (preferred)
                if grep -q "Aggregate Trust Monitor Resource Usage" $output_file; then
                    echo "      📊 Aggregate across all honest nodes:"
                    
                    local avg_cpu=$(grep "Average CPU usage across.*honest nodes:" $output_file | grep -o '[0-9]*\.[0-9]*%' || echo "N/A")
                    local avg_memory=$(grep "Average memory usage across.*honest nodes:" $output_file | grep -o '[0-9]*\.[0-9]*MB' || echo "N/A") 
                    local total_processing=$(grep "Total processing time across all honest nodes:" $output_file | grep -o '[0-9]*\.[0-9]*ms' || echo "N/A")
                    local total_operations=$(grep "Total trust operations across all honest nodes:" $output_file | grep -o '[0-9]* operations' || echo "N/A")
                    
                    echo "        Average CPU: $avg_cpu"
                    echo "        Average Memory: $avg_memory"
                    echo "        Total Processing: $total_processing"
                    echo "        Total Operations: $total_operations"
                else
                    echo "      📊 First node sample (individual):"
                    
                    # Extract first node's usage as sample
                    local cpu_usage=$(grep -A 10 "trust monitor resource usage:" $output_file | grep "CPU:" | head -1 | grep -o '[0-9]*\.[0-9]*% avg' || echo "N/A")
                    local memory_usage=$(grep -A 10 "trust monitor resource usage:" $output_file | grep "Memory:" | head -1 | grep -o '[0-9]*\.[0-9]*MB avg' || echo "N/A")
                    local processing_time=$(grep -A 10 "trust monitor resource usage:" $output_file | grep "Processing:" | head -1 | grep -o '[0-9]*\.[0-9]*ms total' || echo "N/A")
                    local measurements=$(grep -A 10 "trust monitor resource usage:" $output_file | grep "Measurements:" | head -1 | grep -o '[0-9]* operations' || echo "N/A")
                    
                    echo "        CPU: $cpu_usage"
                    echo "        Memory: $memory_usage"
                    echo "        Processing: $processing_time"
                    echo "        Operations tracked: $measurements"
                fi
            fi
            
            if [ "$trust_resource_found" = "false" ]; then
                echo "      No trust resource monitoring data found (may be disabled)"
            fi
        fi
        
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
    echo "🚀 TRUST vs BASELINE EXPERIMENTS"
    echo "================================="
    
    datasets=("cifar10" "eurosat")
    topologies=("complete" "ring")  # Test dense vs sparse connectivity
    attack_types=("label_flip" "gradient")
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


# Main execution
echo "Starting experiments at: $(date)"
echo "Output directory: $(pwd)"
echo ""

# Run trust vs baseline experiments across topologies and node counts
run_scalability_experiments

echo "=========================================="
echo "All experiments completed at: $(date)"
echo "=========================================="

# Generate comprehensive summary report
echo ""
echo "📋 COMPREHENSIVE SUMMARY REPORT"
echo "==============================="

# Create CSV summary file
csv_file="experiment_summary.csv"
echo "Dataset,Topology,Attack,Nodes,Trust,Actual_Malicious,Detected_Malicious,Detection_Count,Detection_Round,Initial_Accuracy,Final_Accuracy,Accuracy_Improvement,Aggregation_Strategy,Mixing_Parameter,Trust_Weighted_Calls,Baseline_Calls,Trust_CPU_Avg_Pct,Trust_Memory_Avg_MB,Trust_Processing_Total_MS,Trust_Operations_Total,Status" > $csv_file

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
            actual_malicious=$(grep "Malicious clients will be created at indices:" $file | grep -o "\[.*\]" | head -1 | tr -d '[]' | tr -d ' ' || echo "N/A")
            
            # Detection results - updated for new log format
            if [ "$trust_enabled" = "true" ]; then
                # Extract detected suspicious neighbors from the global detection message
                detected_line=$(grep "Trust monitoring detected.*suspicious neighbors across all nodes:" $file | tail -1)
                if [ -n "$detected_line" ]; then
                    # Extract node numbers from the format: ['node_1', 'node_0', 'node_7']
                    detected_nodes=$(echo "$detected_line" | grep -o "\['[^']*'[^]]*\]" | sed "s/'node_/'/g" | sed "s/'//g" | sed 's/\[//' | sed 's/\]//' | tr ',' '\n' | sed 's/^ *//' | sort -n | uniq)
                    if [ -n "$detected_nodes" ]; then
                        detected_malicious=$(echo "$detected_nodes" | tr '\n' ',' | sed 's/,$//')
                        detection_count=$(echo $detected_nodes | wc -w | tr -d ' ')
                    else
                        detected_malicious="None"
                        detection_count="0"
                    fi
                else
                    detected_malicious="None"
                    detection_count="0"
                fi
                # First detection round from global detection message
                detection_round=$(grep -B 10 "Trust monitoring detected.*suspicious neighbors across all nodes:" $file | grep "Round [0-9]" | tail -1 | grep -o "[0-9]*" || echo "N/A")
            else
                detected_malicious="N/A"
                detection_count="N/A"
                detection_round="N/A"
            fi
            
            # Accuracy results - fixed initial accuracy parsing (no "Honest Nodes" suffix)
            initial_acc=$(grep "Initial Test Accuracy:" $file | tail -1 | grep -o "[0-9]*\.[0-9]*%" || echo "N/A")
            final_acc=$(grep "Final Test Accuracy.*Honest Nodes" $file | tail -1 | grep -o "[0-9]*\.[0-9]*%" || grep "Final Test Accuracy:" $file | tail -1 | grep -o "[0-9]*\.[0-9]*%" || echo "N/A")
            acc_improvement=$(grep "Accuracy Improvement.*Honest Nodes" $file | tail -1 | grep -o "[-]*[0-9]*\.[0-9]*%" || grep "Accuracy Improvement:" $file | tail -1 | grep -o "[-]*[0-9]*\.[0-9]*%" || echo "N/A")
            
            # Extract aggregation strategy information for CSV
            aggregation_strategy=$(grep "AGGREGATION_DEBUG.*Round 1:" $file | head -1 | grep -o "strategy=[A-Za-z]*" | sed 's/strategy=//' || echo "Unknown")
            mixing_parameter=$(grep "AGGREGATION_DEBUG.*Round 1:" $file | head -1 | grep -o "mixing=[0-9.]*" | sed 's/mixing=//' || echo "N/A")
            trust_weighted_calls=$(grep -c "TRUST-WEIGHTED aggregation" $file || echo "0")
            baseline_calls=$(grep -c "BASELINE aggregation" $file || echo "0")
            
            echo "   🎯 Actual malicious: [$actual_malicious]"
            echo "   🔍 Detected malicious: [$detected_malicious]"
            echo "   📍 Detection: $detection_count malicious nodes in round $detection_round"
            echo "   🚀 Initial Accuracy: $initial_acc"
            echo "   🎯 Final Accuracy: $final_acc"
            echo "   📈 Accuracy Improvement: $acc_improvement"
            echo "   🔄 Strategy: $aggregation_strategy (mixing: $mixing_parameter)"
            echo "   📊 Aggregation calls: Trust=$trust_weighted_calls, Baseline=$baseline_calls"
            echo "   📄 File: $file"
            
            # Extract trust resource metrics for trust-enabled experiments
            trust_cpu_avg="N/A"
            trust_memory_avg="N/A" 
            trust_processing_total="N/A"
            trust_operations="N/A"
            
            if [ "$trust_enabled" = "true" ]; then
                # Try to extract aggregate values first (preferred), fallback to individual node values
                trust_cpu_avg=$(grep "Average CPU usage across.*honest nodes:" $file | grep -o '[0-9]*\.[0-9]*' || grep -A 10 "trust monitor resource usage:" $file | grep "CPU:" | head -1 | grep -o '[0-9]*\.[0-9]*' || echo "N/A")
                trust_memory_avg=$(grep "Average memory usage across.*honest nodes:" $file | grep -o '[0-9]*\.[0-9]*' || grep -A 10 "trust monitor resource usage:" $file | grep "Memory:" | head -1 | grep -o '[0-9]*\.[0-9]*' || echo "N/A")
                trust_processing_total=$(grep "Total processing time across all honest nodes:" $file | grep -o '[0-9]*\.[0-9]*' || grep -A 10 "trust monitor resource usage:" $file | grep "Processing:" | head -1 | grep -o '[0-9]*\.[0-9]*' || echo "N/A")
                trust_operations=$(grep "Total trust operations across all honest nodes:" $file | grep -o '[0-9]*' || grep -A 10 "trust monitor resource usage:" $file | grep "Measurements:" | head -1 | grep -o '[0-9]*' || echo "N/A")
            fi
            
            # Add to CSV with new aggregation strategy fields
            echo "$dataset,$topology,$attack,$node_count,$trust_enabled,$actual_malicious,$detected_malicious,$detection_count,$detection_round,$initial_acc,$final_acc,$acc_improvement,$aggregation_strategy,$mixing_parameter,$trust_weighted_calls,$baseline_calls,$trust_cpu_avg,$trust_memory_avg,$trust_processing_total,$trust_operations,$status" >> $csv_file
        else
            status="FAILED"
            echo "   ❌ FAILED - Check $file for details"
            echo "$dataset,$topology,$attack,$node_count,$trust_enabled,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,N/A,$status" >> $csv_file
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
echo "   grep 'Trust monitoring detected.*suspicious neighbors across all nodes' *.txt | sort"
echo ""
echo "   # Aggregation strategy verification:"
echo "   echo \"Trust-weighted strategy usage:\"; grep 'AGGREGATION_DEBUG.*TRUST-WEIGHTED' *_trust.txt | wc -l"
echo "   echo \"Baseline strategy usage:\"; grep 'AGGREGATION_DEBUG.*BASELINE' *_baseline.txt | wc -l"
echo "   echo \"Mixing parameters used:\"; grep 'AGGREGATION_DEBUG.*Round 1:' *.txt | grep -o 'mixing=[0-9.]*'"
echo ""
echo "   # Trust score analysis (trust-weighted experiments only):"
echo "   echo \"Trust score evolution samples:\"; grep 'TRUST_WEIGHTED_GOSSIP: Trust scores:' *_trust.txt | head -5"
echo "   echo \"Zero trust instances:\"; grep 'All neighbors have zero trust' *_trust.txt | wc -l"
echo ""
echo "   # Trust evolution analysis:"
echo "   echo \"Trust evolution files:\"; ls -la *_trust_evolution.txt 2>/dev/null || echo \"None found\""
echo ""
echo "   # Scalability analysis:"
echo "   for n in 20 30 50; do echo \"Node count \$n:\"; grep \"Final Test Accuracy\" *n\$n*.txt; done"
echo ""
echo "   # Resource usage comparison (trust vs baseline):"
echo "   echo \"Trust-enabled resource usage:\"; grep \"resource usage:\" *_trust.txt | head -5"
echo "   echo \"Baseline resource usage:\"; grep \"resource usage:\" *_baseline.txt | head -5"
echo ""
echo "   # Trust monitor overhead analysis:"
echo "   echo \"Trust resource monitoring summaries:\"; grep -A 15 \"Trust Monitor Resource Usage Summary\" *_trust.txt 2>/dev/null | head -20 || echo \"No trust resource data found\""
echo "   echo \"Trust monitoring CPU usage:\"; grep \"CPU:.*avg\" *_trust.txt 2>/dev/null || echo \"No trust CPU data found\""
echo "   echo \"Trust monitoring memory usage:\"; grep \"Memory:.*avg\" *_trust.txt 2>/dev/null || echo \"No trust memory data found\""
echo ""
echo "   # Aggregation debugging (new logs):"
echo "   echo \"Aggregation strategy breakdown by experiment:\"; grep 'AGGREGATION_DEBUG.*Round 1:' *.txt | cut -d: -f1,4 | sort"
echo "   echo \"Sample trust weight distributions:\"; grep 'Normalized neighbor weights:' *_trust.txt | head -3"
echo "   echo \"Sample baseline weight distributions:\"; grep 'Base weights:' *_baseline.txt | head -3"
echo ""
echo "=========================================="