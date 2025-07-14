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
"

# Trust monitoring parameters
TRUST_PARAMS="
    --enable_trust_monitoring
    --enable_trust_weighted_aggregation
    --enable_exponential_decay
    --exponential_decay_base 0.6
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
        
        echo "   🎯 Final accuracy: $(grep "Final Test Accuracy" $output_file | tail -1 || echo "Not found")"
        echo "   📊 Accuracy improvement: $(grep "Accuracy Improvement" $output_file | tail -1 || echo "Not found")"
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
            
            # Accuracy results  
            final_acc=$(grep "Final Test Accuracy" $file | tail -1 | grep -o "[0-9]*\.[0-9]*%" || echo "N/A")
            acc_improvement=$(grep "Accuracy Improvement" $file | tail -1 | grep -o "[-]*[0-9]*\.[0-9]*%" || echo "N/A")
            
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
echo ""
echo "🔍 Quick Analysis Commands:"
echo ""
echo "   # Compare trust vs baseline accuracy:"
echo "   grep 'Final Test Accuracy' *_trust.txt | sort"
echo "   grep 'Final Test Accuracy' *_baseline.txt | sort"
echo ""
echo "   # Detection performance across node counts:"
echo "   grep 'Trust monitoring detected.*suspicious neighbors' *.txt | sort"
echo ""
echo "   # Scalability analysis:"
echo "   for n in 10 20 30 50; do echo \"Node count \$n:\"; grep \"Final Test Accuracy\" *n\$n*.txt; done"
echo ""
echo "📊 For detailed analysis, run:"
echo "   python ../analyze_enhanced_results.py $OUTPUT_DIR"
echo ""
echo "=========================================="