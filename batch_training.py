import os
import sys
import numpy as np
import time
from time import gmtime, strftime
import torch
import torch.nn as nn
import torch.optim as optim
from IPython import embed

import parameters
from data_loader import UnifiedDataGenerator
from seld_result import ComputeSELDResults, reshape_3Dto2D
from evaluation_metrics import distance_between_cartesian_coordinates
import seldnet_model


def get_multi_accdoa_labels(accdoa_in, nb_classes):
    x0, y0, z0 = accdoa_in[:, :, :1*nb_classes], accdoa_in[:, :, 1*nb_classes:2*nb_classes], accdoa_in[:, :, 2*nb_classes:3*nb_classes]
    dist0 = accdoa_in[:, :, 3*nb_classes:4*nb_classes]
    dist0[dist0 < 0.] = 0.
    sed0 = np.sqrt(x0**2 + y0**2 + z0**2) > 0.5
    doa0 = accdoa_in[:, :, :3*nb_classes]

    x1, y1, z1 = accdoa_in[:, :, 4*nb_classes:5*nb_classes], accdoa_in[:, :, 5*nb_classes:6*nb_classes], accdoa_in[:, :, 6*nb_classes:7*nb_classes]
    dist1 = accdoa_in[:, :, 7*nb_classes:8*nb_classes]
    dist1[dist1 < 0.] = 0.
    sed1 = np.sqrt(x1**2 + y1**2 + z1**2) > 0.5
    doa1 = accdoa_in[:, :, 4*nb_classes:7*nb_classes]

    x2, y2, z2 = accdoa_in[:, :, 8*nb_classes:9*nb_classes], accdoa_in[:, :, 9*nb_classes:10*nb_classes], accdoa_in[:, :, 10*nb_classes:11*nb_classes]
    dist2 = accdoa_in[:, :, 11*nb_classes:]
    dist2[dist2 < 0.] = 0.
    sed2 = np.sqrt(x2**2 + y2**2 + z2**2) > 0.5
    doa2 = accdoa_in[:, :, 8*nb_classes:11*nb_classes]

    return sed0, doa0, dist0, sed1, doa1, dist1, sed2, doa2, dist2


def determine_similar_location(sed_pred0, sed_pred1, doa_pred0, doa_pred1, class_cnt, thresh_unify, nb_classes):
    """Determine if two tracks are similar enough to unify"""
    if (sed_pred0 == 1) and (sed_pred1 == 1):
        if distance_between_cartesian_coordinates(
            doa_pred0[class_cnt], doa_pred0[class_cnt+1*nb_classes], doa_pred0[class_cnt+2*nb_classes],
            doa_pred1[class_cnt], doa_pred1[class_cnt+1*nb_classes], doa_pred1[class_cnt+2*nb_classes]
        ) < thresh_unify:
            return 1
        else:
            return 0
    else:
        return 0


def write_predictions_to_file(output_dict, output_file, data_generator):
    data_generator.write_output_format_file(output_file, output_dict)


def validate_epoch(data_generator, model, criterion, dcase_output_folder, params, device):
    nb_val_batches, val_loss = 0, 0.
    model.eval()
    
    with torch.no_grad():
        for features, targets in data_generator.generate():
            # Move to device
            features = torch.tensor(features).to(device).float()
            targets = torch.tensor(targets).to(device).float()
            
            # Forward pass
            output = model(features)
            loss = criterion(output, targets)
            
            # Update loss
            val_loss += loss.item()
            nb_val_batches += 1
            
            if params['quick_test'] and nb_val_batches == 4:
                break
        
        val_loss /= nb_val_batches if nb_val_batches > 0 else 1
    
    return val_loss


def test_epoch(data_generator, model, criterion, dcase_output_folder, params, device, mode='test'):
    """Test epoch - computes loss and writes predictions"""
    test_filelist = data_generator.get_filelist()
    
    nb_test_batches, test_loss = 0, 0.
    model.eval()
    file_cnt = 0
    
    with torch.no_grad():
        for features, targets in data_generator.generate():
            # Move to device
            features = torch.tensor(features).to(device).float()
            targets = torch.tensor(targets).to(device).float()
            
            # Forward pass
            output = model(features)
            loss = criterion(output, targets)
            
            # Convert predictions to numpy for processing
            output_np = output.detach().cpu().numpy()
            
            # Get predictions (multi-ACCDOA always True)
            sed_pred0, doa_pred0, dist_pred0, sed_pred1, doa_pred1, dist_pred1, sed_pred2, doa_pred2, dist_pred2 = get_multi_accdoa_labels(
                output_np, params['unique_classes']
            )
            
            # Reshape to 2D for easier processing
            sed_pred0 = reshape_3Dto2D(sed_pred0)
            doa_pred0 = reshape_3Dto2D(doa_pred0)
            dist_pred0 = reshape_3Dto2D(dist_pred0)
            sed_pred1 = reshape_3Dto2D(sed_pred1)
            doa_pred1 = reshape_3Dto2D(doa_pred1)
            dist_pred1 = reshape_3Dto2D(dist_pred1)
            sed_pred2 = reshape_3Dto2D(sed_pred2)
            doa_pred2 = reshape_3Dto2D(doa_pred2)
            dist_pred2 = reshape_3Dto2D(dist_pred2)
            
            # Create output dictionary for each file in the batch
            # Since we're in per_file mode for validation/test, batch_size = 1
            for batch_idx in range(features.shape[0]):
                if file_cnt >= len(test_filelist):
                    break
                    
                output_dict = {}
                for frame_cnt in range(sed_pred0.shape[0]):
                    frame_idx = frame_cnt
                    for class_cnt in range(params['unique_classes']):
                        flag_0sim1 = determine_similar_location(
                            sed_pred0[batch_idx, frame_idx, class_cnt], 
                            sed_pred1[batch_idx, frame_idx, class_cnt],
                            doa_pred0[batch_idx, frame_idx], 
                            doa_pred1[batch_idx, frame_idx], 
                            class_cnt,
                            params['thresh_unify'], 
                            params['unique_classes']
                        )
                        flag_1sim2 = determine_similar_location(
                            sed_pred1[batch_idx, frame_idx, class_cnt], 
                            sed_pred2[batch_idx, frame_idx, class_cnt],
                            doa_pred1[batch_idx, frame_idx], 
                            doa_pred2[batch_idx, frame_idx], 
                            class_cnt,
                            params['thresh_unify'], 
                            params['unique_classes']
                        )
                        flag_2sim0 = determine_similar_location(
                            sed_pred2[batch_idx, frame_idx, class_cnt], 
                            sed_pred0[batch_idx, frame_idx, class_cnt],
                            doa_pred2[batch_idx, frame_idx], 
                            doa_pred0[batch_idx, frame_idx], 
                            class_cnt,
                            params['thresh_unify'], 
                            params['unique_classes']
                        )
                        
                        # Unify or not unify according to flags
                        if flag_0sim1 + flag_1sim2 + flag_2sim0 == 0:
                            # No unification needed
                            if sed_pred0[batch_idx, frame_idx, class_cnt] > 0.5:
                                if frame_idx not in output_dict:
                                    output_dict[frame_idx] = []
                                output_dict[frame_idx].append([
                                    class_cnt,
                                    doa_pred0[batch_idx, frame_idx, class_cnt],
                                    doa_pred0[batch_idx, frame_idx, class_cnt + params['unique_classes']],
                                    doa_pred0[batch_idx, frame_idx, class_cnt + 2 * params['unique_classes']],
                                    dist_pred0[batch_idx, frame_idx, class_cnt]
                                ])
                            if sed_pred1[batch_idx, frame_idx, class_cnt] > 0.5:
                                if frame_idx not in output_dict:
                                    output_dict[frame_idx] = []
                                output_dict[frame_idx].append([
                                    class_cnt,
                                    doa_pred1[batch_idx, frame_idx, class_cnt],
                                    doa_pred1[batch_idx, frame_idx, class_cnt + params['unique_classes']],
                                    doa_pred1[batch_idx, frame_idx, class_cnt + 2 * params['unique_classes']],
                                    dist_pred1[batch_idx, frame_idx, class_cnt]
                                ])
                            if sed_pred2[batch_idx, frame_idx, class_cnt] > 0.5:
                                if frame_idx not in output_dict:
                                    output_dict[frame_idx] = []
                                output_dict[frame_idx].append([
                                    class_cnt,
                                    doa_pred2[batch_idx, frame_idx, class_cnt],
                                    doa_pred2[batch_idx, frame_idx, class_cnt + params['unique_classes']],
                                    doa_pred2[batch_idx, frame_idx, class_cnt + 2 * params['unique_classes']],
                                    dist_pred2[batch_idx, frame_idx, class_cnt]
                                ])
                        elif flag_0sim1 + flag_1sim2 + flag_2sim0 == 1:
                            # Unify two similar tracks
                            if frame_idx not in output_dict:
                                output_dict[frame_idx] = []
                            
                            if flag_0sim1:
                                # Keep track 2, unify 0 and 1
                                if sed_pred2[batch_idx, frame_idx, class_cnt] > 0.5:
                                    output_dict[frame_idx].append([
                                        class_cnt,
                                        doa_pred2[batch_idx, frame_idx, class_cnt],
                                        doa_pred2[batch_idx, frame_idx, class_cnt + params['unique_classes']],
                                        doa_pred2[batch_idx, frame_idx, class_cnt + 2 * params['unique_classes']],
                                        dist_pred2[batch_idx, frame_idx, class_cnt]
                                    ])
                                doa_pred_fc = (doa_pred0[batch_idx, frame_idx] + doa_pred1[batch_idx, frame_idx]) / 2
                                dist_pred_fc = (dist_pred0[batch_idx, frame_idx] + dist_pred1[batch_idx, frame_idx]) / 2
                                output_dict[frame_idx].append([
                                    class_cnt,
                                    doa_pred_fc[class_cnt],
                                    doa_pred_fc[class_cnt + params['unique_classes']],
                                    doa_pred_fc[class_cnt + 2 * params['unique_classes']],
                                    dist_pred_fc[class_cnt]
                                ])
                            elif flag_1sim2:
                                # Keep track 0, unify 1 and 2
                                if sed_pred0[batch_idx, frame_idx, class_cnt] > 0.5:
                                    output_dict[frame_idx].append([
                                        class_cnt,
                                        doa_pred0[batch_idx, frame_idx, class_cnt],
                                        doa_pred0[batch_idx, frame_idx, class_cnt + params['unique_classes']],
                                        doa_pred0[batch_idx, frame_idx, class_cnt + 2 * params['unique_classes']],
                                        dist_pred0[batch_idx, frame_idx, class_cnt]
                                    ])
                                doa_pred_fc = (doa_pred1[batch_idx, frame_idx] + doa_pred2[batch_idx, frame_idx]) / 2
                                dist_pred_fc = (dist_pred1[batch_idx, frame_idx] + dist_pred2[batch_idx, frame_idx]) / 2
                                output_dict[frame_idx].append([
                                    class_cnt,
                                    doa_pred_fc[class_cnt],
                                    doa_pred_fc[class_cnt + params['unique_classes']],
                                    doa_pred_fc[class_cnt + 2 * params['unique_classes']],
                                    dist_pred_fc[class_cnt]
                                ])
                            elif flag_2sim0:
                                # Keep track 1, unify 2 and 0
                                if sed_pred1[batch_idx, frame_idx, class_cnt] > 0.5:
                                    output_dict[frame_idx].append([
                                        class_cnt,
                                        doa_pred1[batch_idx, frame_idx, class_cnt],
                                        doa_pred1[batch_idx, frame_idx, class_cnt + params['unique_classes']],
                                        doa_pred1[batch_idx, frame_idx, class_cnt + 2 * params['unique_classes']],
                                        dist_pred1[batch_idx, frame_idx, class_cnt]
                                    ])
                                doa_pred_fc = (doa_pred2[batch_idx, frame_idx] + doa_pred0[batch_idx, frame_idx]) / 2
                                dist_pred_fc = (dist_pred2[batch_idx, frame_idx] + dist_pred0[batch_idx, frame_idx]) / 2
                                output_dict[frame_idx].append([
                                    class_cnt,
                                    doa_pred_fc[class_cnt],
                                    doa_pred_fc[class_cnt + params['unique_classes']],
                                    doa_pred_fc[class_cnt + 2 * params['unique_classes']],
                                    dist_pred_fc[class_cnt]
                                ])
                        else:
                            # Unify all three tracks
                            if frame_idx not in output_dict:
                                output_dict[frame_idx] = []
                            doa_pred_fc = (doa_pred0[batch_idx, frame_idx] + doa_pred1[batch_idx, frame_idx] + doa_pred2[batch_idx, frame_idx]) / 3
                            dist_pred_fc = (dist_pred0[batch_idx, frame_idx] + dist_pred1[batch_idx, frame_idx] + dist_pred2[batch_idx, frame_idx]) / 3
                            output_dict[frame_idx].append([
                                class_cnt,
                                doa_pred_fc[class_cnt],
                                doa_pred_fc[class_cnt + params['unique_classes']],
                                doa_pred_fc[class_cnt + 2 * params['unique_classes']],
                                dist_pred_fc[class_cnt]
                            ])
                
                # Write predictions to file
                output_file = os.path.join(dcase_output_folder, test_filelist[file_cnt].replace('.npy', '.csv'))
                write_predictions_to_file(output_dict, output_file, data_generator)
                file_cnt += 1
            
            # Update loss
            test_loss += loss.item()
            nb_test_batches += 1
            
            if params['quick_test'] and nb_test_batches == 4:
                break
        
        test_loss /= nb_test_batches if nb_test_batches > 0 else 1
    
    return test_loss


def train_epoch(data_generator, optimizer, model, criterion, params, device):
    nb_train_batches, train_loss = 0, 0.
    model.train()
    
    for features, targets in data_generator.generate():
        # Move to device
        features = torch.tensor(features).to(device).float()
        targets = torch.tensor(targets).to(device).float()
        
        # Forward pass
        optimizer.zero_grad()
        output = model(features)
        loss = criterion(output, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update statistics
        train_loss += loss.item()
        nb_train_batches += 1
        
        if params['quick_test'] and nb_train_batches == 4:
            break
    
    train_loss /= nb_train_batches if nb_train_batches > 0 else 1
    return train_loss


def main(argv):
    print("Starting training with arguments:", argv)
    
    # Set up device
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.autograd.set_detect_anomaly(True)
    
    # Get parameters
    params = parameters.params
    
    job_id = 1 if len(argv) < 3 else argv[-1]
    
    # data split
    test_splits = [[4]]
    val_splits = [[4]]
    train_splits = [[3]]
        
    for split_cnt, split in enumerate(test_splits):
        print('\n\n' + '='*80)
        print(f'Training on split {split}')
        print('='*80)
        
        model_name = 'multi_accdoa'
        print(f"Model will be saved as: {model_name}")
        
        os.makedirs(params['model_dir'], exist_ok=True)
        
        # Load training data
        print('\nLoading training dataset...')
        data_gen_train = UnifiedDataGenerator(
            params=params, 
            split=train_splits[split_cnt],
            shuffle=True,
            per_file=False
        )
        
        # Load validation data
        print('Loading validation dataset...')
        data_gen_val = UnifiedDataGenerator(
            params=params,
            split=val_splits[split_cnt],
            shuffle=False,
            per_file=True
        )
        
        # Get data sizes and create model
        data_in, data_out = data_gen_train.get_data_sizes()
        print(f'\nData shapes - Input: {data_in}, Output: {data_out}')
        
        # Create model
        model = seldnet_model.SeldModel(data_in, data_out, params).to(device)
        
        # Print model architecture
        print('\nModel architecture:')
        print(model)
        
        # Create output folder for validation results
        dcase_output_val_folder = os.path.join(
            params['dcase_output_dir'],
            f'{unique_name}_val_{strftime("%Y%m%d%H%M%S", gmtime())}'
        )
        os.makedirs(dcase_output_val_folder, exist_ok=True)
        print(f'\nValidation results will be saved in: {dcase_output_val_folder}')
        
        # Initialize evaluation metric class
        score_obj = ComputeSELDResults(params)
        
        # Training setup
        best_val_epoch = -1
        best_val_loss = float('inf')
        best_F, best_LE, best_seld_scr = 0., 180., 9999.
        patience_cnt = 0
        
        nb_epoch = 2 if params['quick_test'] else params['nb_epochs']
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        
        # Use MSELoss_ADPIT for multi-ACCDOA
        criterion = seldnet_model.MSELoss_ADPIT()
        
        print(f'\nStarting training for {nb_epoch} epochs...')
        print('-'*80)
        
        for epoch_cnt in range(nb_epoch):
            # Training phase
            start_time = time.time()
            train_loss = train_epoch(data_gen_train, optimizer, model, criterion, params, device)
            train_time = time.time() - start_time
            
            # Validation phase (compute loss only)
            start_time = time.time()
            val_loss = validate_epoch(data_gen_val, model, criterion, dcase_output_val_folder, params, device)
            val_time = time.time() - start_time
            
            # Periodically compute full metrics (e.g., every 5 epochs or at the end)
            if epoch_cnt % 5 == 0 or epoch_cnt == nb_epoch - 1:
                # Run test epoch to get predictions and compute metrics
                test_loss = test_epoch(data_gen_val, model, criterion, dcase_output_val_folder, params, device, mode='val')
                
                # Calculate metrics
                val_ER, val_F, val_LE, val_dist_err, val_rel_dist_err, val_LR, val_seld_scr, _ = score_obj.get_SELD_Results(
                    dcase_output_val_folder
                )
                
                # Save best model based on validation loss
                if val_loss < best_val_loss:
                    best_val_epoch = epoch_cnt
                    best_val_loss = val_loss
                    best_F = val_F
                    best_LE = val_LE
                    best_seld_scr = val_seld_scr
                    torch.save(model.state_dict(), model_name)
                    patience_cnt = 0
                    print(f'  -> New best model saved! (val_loss={val_loss:.4f})')
                else:
                    patience_cnt += 1
                
                # Print epoch statistics
                print(f'Epoch {epoch_cnt:3d}: '
                      f'train_loss={train_loss:.4f}, '
                      f'val_loss={val_loss:.4f}, '
                      f'F={val_F:.3f}, '
                      f'AE={val_LE:.1f}°, '
                      f'SELD={val_seld_scr:.3f}, '
                      f'time={train_time+val_time:.1f}s '
                      f'[best: epoch={best_val_epoch}, val_loss={best_val_loss:.4f}, F={best_F:.3f}, AE={best_LE:.1f}°, SELD={best_seld_scr:.3f}]')
            else:
                # Just print losses
                print(f'Epoch {epoch_cnt:3d}: '
                      f'train_loss={train_loss:.4f}, '
                      f'val_loss={val_loss:.4f}, '
                      f'time={train_time+val_time:.1f}s')
            
            # Early stopping
            if patience_cnt > params['patience']:
                print(f'\nEarly stopping triggered after {epoch_cnt+1} epochs')
                break
        
        # Test phase
        print('\n' + '='*80)
        print('Evaluating on test set...')
        print('='*80)
        
        # Load best model
        print('Loading best model weights...')
        model.load_state_dict(torch.load(model_name, map_location=device))
        
        # Load test data
        print('Loading test dataset...')
        data_gen_test = UnifiedDataGenerator(
            params=params,
            split=test_splits[split_cnt],
            shuffle=False,
            per_file=True
        )
        
        # Create output folder for test results
        dcase_output_test_folder = os.path.join(
            params['dcase_output_dir'],
            f'{unique_name}_test_{strftime("%Y%m%d%H%M%S", gmtime())}'
        )
        os.makedirs(dcase_output_test_folder, exist_ok=True)
        print(f'Test results will be saved in: {dcase_output_test_folder}')
        
        # Run test
        test_loss = test_epoch(data_gen_test, model, criterion, dcase_output_test_folder, params, device, mode='test')
        
        # Calculate test metrics
        use_jackknife = True
        test_ER, test_F, test_LE, test_dist_err, test_rel_dist_err, test_LR, test_seld_scr, classwise_test_scr = score_obj.get_SELD_Results(
            dcase_output_test_folder, is_jackknife=use_jackknife
        )
        
        # Print final results
        print('\n' + '='*80)
        print('FINAL TEST RESULTS:')
        print('='*80)
        print(f'SELD score: {test_seld_scr[0]:.3f} ' + 
              (f'[{test_seld_scr[1][0]:.3f}, {test_seld_scr[1][1]:.3f}]' if use_jackknife else ''))
        print(f'SED F-score: {100*test_F[0]:.1f}% ' +
              (f'[{100*test_F[1][0]:.1f}%, {100*test_F[1][1]:.1f}%]' if use_jackknife else ''))
        print(f'DOA Angular Error: {test_LE[0]:.1f}° ' +
              (f'[{test_LE[1][0]:.1f}°, {test_LE[1][1]:.1f}°]' if use_jackknife else ''))
        print(f'Distance Error: {test_dist_err[0]:.3f} ' +
              (f'[{test_dist_err[1][0]:.3f}, {test_dist_err[1][1]:.3f}]' if use_jackknife else ''))
        
        # Print class-wise results if available
        if params['average'] == 'macro' and classwise_test_scr is not None:
            print('\nClass-wise results:')
            print('Class\tF-score\tAngular Error\tDistance Error\tSELD Score')
            for cls_cnt in range(params['unique_classes']):
                if use_jackknife:
                    f_score = classwise_test_scr[0][1][cls_cnt]
                    ang_err = classwise_test_scr[0][2][cls_cnt]
                    dist_err = classwise_test_scr[0][3][cls_cnt]
                    seld_score = classwise_test_scr[0][6][cls_cnt]
                else:
                    f_score = classwise_test_scr[1][cls_cnt]
                    ang_err = classwise_test_scr[2][cls_cnt]
                    dist_err = classwise_test_scr[3][cls_cnt]
                    seld_score = classwise_test_scr[6][cls_cnt]
                
                print(f'{cls_cnt}\t{f_score:.3f}\t{ang_err:.1f}°\t\t{dist_err:.3f}\t\t{seld_score:.3f}')
        
        print('\n' + '='*80)
        print('Training completed successfully!')
        print('='*80)


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except (ValueError, IOError) as e:
        print(f"Error: {e}")
        sys.exit(1)