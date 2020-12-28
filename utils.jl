# custom functions used in corresponding jupyter notebook

unzip(a) = map(x->getfield.(a, x), [1, 2]);

function convert_positions_to_labels(positions; grid_size=64, image_size=256, classes=false)
    """ Returns 64x64xchannelsxn_images label tensors computed from list of vortex positions """
    n_images = size(positions, 3)
    grid_cell = image_size ÷ grid_size
    max_objects = size(positions, 1)
    if classes channels=4 else channels=3 end
    labels = zeros(grid_size, grid_size, channels, n_images)
    for n = 1:n_images
        for i = 1:max_objects
            if positions[i,1,n] > 0 # check first if object exists
                x = Int((positions[i,1,n] - 1) ÷ grid_cell + 1)
                y = Int((positions[i,2,n] - 1) ÷ grid_cell + 1)

                x_offset = (positions[i,1,n] - 1) / grid_cell + 1 - x
                y_offset = (positions[i,2,n] - 1) / grid_cell + 1 - y

                if classes
                    if positions[i,3,n] == 1.0
                        labels[x, y, 1, n] = 1.0
                    elseif positions[i,3,n] == -1.0
                        labels[x, y, 2, n] = 1.0
                    else
                        error("class does not exist")
                    end
                    labels[x, y, 3, n] = x_offset
                    labels[x, y, 4, n] = y_offset
                else
                    labels[x, y, 1, n] = 1.0
                    labels[x, y, 2, n] = x_offset
                    labels[x, y, 3, n] = y_offset
                end
            end
        end
    end
    return labels
end;

function convert_label_to_position(label; threshold=0.5, grid_cell=4, classes=false)
    """ Returns list of vortex positions computed from single 64x64xchannel label tensor """
    positions, confidence = [], []
    if classes
        thres_min = minimum(threshold)
        combined = max.(label[:,:,1], label[:,:,2])
        grid_idx = findall(x->x>=thres_min, combined[:,:])
        if length(threshold) == 1
            threshold = [threshold, threshold]
        end
    else
        grid_idx = findall(x->x>=threshold, label[:,:,1])
    end
    for (i, id) in enumerate(grid_idx)
        conf = label[id[1],id[2],1]
        grid_offset_x = grid_cell * (id[1] - 1)
        grid_offset_y = grid_cell * (id[2] - 1)
        if classes
            if label[id[1],id[2],1] > label[id[1],id[2],2]
                if label[id[1],id[2],1] >= threshold[1]
                    class = 1.0
                    conf = label[id[1],id[2],1]
                else
                    continue
                end
            else
                if label[id[1],id[2],2] >= threshold[2]
                    class = -1.0
                    conf = label[id[1],id[2],2]
                else
                    continue
                end
            end
            x = grid_offset_x + grid_cell * label[id[1],id[2],3] + 1
            y = grid_offset_y + grid_cell * label[id[1],id[2],4] + 1
            append!(positions, [[round(x), round(y), class]])
        else
            x = grid_offset_x + grid_cell * label[id[1],id[2],2] + 1
            y = grid_offset_y + grid_cell * label[id[1],id[2],3] + 1
            append!(positions, [[round(x), round(y)]])
        end
        append!(confidence, conf)
    end
    return round.(Int, hcat(positions...)'), confidence
end;

function convert_labels_to_positions(labels; threshold=0.5, grid_cell=4, classes=false)
    """ Returns list of vortex positions computed from a batch of 64x64xchannelsxn_images label tensors """
    return unzip(convert_label_to_position.(eachslice(labels, dims=4), threshold=threshold, grid_cell=grid_cell, classes=classes))
end;

function split_data(images, labels; train_test_ratio=0.8, seed=1234)
    """ Returns training and test set by randomly splitting images into two subsets """
    Random.seed!(seed)
    n_training = Int(n_images*train_test_ratio);

    # shuffle images first before splitting into train and test data
    id = randperm(n_images)
    images = images[:,:,:,id]
    labels = labels[:,:,:,id]

    train_images = images[:,:,:,1:n_training];
    test_images = images[:,:,:,(n_training+1):end];
    train_labels = labels[:,:,:,1:n_training];
    test_labels = labels[:,:,:,(n_training+1):end];

    return train_images, train_labels, test_images, test_labels
end;

function make_minibatch(x, y, idxs)
    """ Returns minibatches by splitting training data according to a list of indices """
    x_batch = Array{Float32}(undef, size(x[:,:,:,1])..., length(idxs))
    y_batch = Array{Float32}(undef, size(y[:,:,:,1])..., length(idxs))
    for i in 1:length(idxs)
        x_batch[:, :, :, i] = Float32.(x[:,:,:,idxs[i]]);
        y_batch[:, :, :, i] = Float32.(y[:,:,:,idxs[i]]);
    end
    return (x_batch, y_batch)
end;

function nms(probability_map; kernel=3, classes=false)
    """ Non-max suppression to eliminate multiple detections of the same vortex """
    pad = (kernel - 1) ÷ 2
    if classes
        pmax = NNlib.maxpool(probability_map[:,:,1:2,:], (kernel, kernel), pad=pad, stride=1)|>gpu
        keep = ones(size(probability_map))|>gpu
        keep[:,:,1:2,:] = float.((pmax .== probability_map[:,:,1:2,:]))
    else
        pmax = NNlib.maxpool(probability_map[:,:,1:1,:], (kernel, kernel), pad=pad, stride=1)|>gpu
        keep = float.((pmax .== probability_map[:,:,1:1,:]))
    end
    return probability_map .* keep
end;

function train(loss, opt, epochs, model, train_set, test_images, test_labels)
    """ Train model for number of epochs, returns trained model and list of test losses """
    losses = []
    test_loss = loss(model, test_images, test_labels)
    println("Loss: $(round(test_loss, digits=2))")
    for epoch = 1:epochs
        for d in train_set
            gs = gradient(Flux.params(model)) do
                l = loss(model, d...)
            end
            update!(opt, Flux.params(model), gs)
        end
        test_loss = loss(model, test_images, test_labels)
        append!(losses, test_loss)
        println("$(epoch), Loss: $(round(test_loss, digits=2))")
    end
    return model, losses
end;

distance(xA, xB, yA, yB) = sqrt((xA - xB)^2 + (yA - yB)^2)

""" Returns F1 score: harmonic mean of precision and recall """
f1score(precision, recall) = 2 .* precision .* recall ./ (precision .+ recall)

function precision_and_recall(y_hat, y; confidence_threshold=0.5, distance_threshold=2.3)
    """ Returns precision and recall for a single image """
    predicted_pos, confidence = convert_label_to_position(y_hat, threshold=confidence_threshold)
    true_pos, _ = convert_label_to_position(y)

    if size(predicted_pos,2) == 0# case 1: no predictions
        precision = 1.0
        if size(true_pos,2) == 0 # case 1a: no objects
            recall = 1.0
        else                     # case 1b: there are objects
            recall = 0.0
        end

    else                         # case 2: there are predictions
        if size(true_pos,2) == 0 # case 2a: no objects
            recall = 1.0
            precision = 0.0
        else                     # case 2b: there are objects
            already_detected = zeros(size(true_pos, 1))
            id = sortperm(-confidence)
            predicted_pos = predicted_pos[id,:]

            true_positives = 0
            false_positives = 0
            for i = 1:size(predicted_pos, 1)
                xB = predicted_pos[i,1]
                yB = predicted_pos[i,2]
                distance_list = zeros(size(true_pos, 1))

                for j = 1:size(true_pos, 1)
                    xA = true_pos[j,1]
                    yA = true_pos[j,2]
                    distance_list[j] = distance(xA, xB, yA, yB)
                end

                jmin = argmin(distance_list)
                dismin = distance_list[jmin]
                if dismin <= distance_threshold # true positives
                    if already_detected[jmin] == 0
                        true_positives += 1
                        already_detected[jmin] = 1
                    else
                        false_positives += 1
                    end
                else
                    false_positives += 1
                end
            end
            total_positives = size(predicted_pos, 1) # same as: true_positives + false_positives
            total_objects = size(true_pos, 1)

            precision = true_positives / total_positives
            recall = true_positives / total_objects
        end
    end
    return precision, recall
end;

function precision_and_recall_classes(y_hat, y; confidence_threshold=0.5, distance_threshold=2.3)
    """ Returns precision and recall for each different class for a single image """
    predicted_pos, confidence = convert_label_to_position(y_hat, threshold=confidence_threshold, classes=true)
    true_pos, _ = convert_label_to_position(y, classes=true)

    if size(predicted_pos,2) == 0 # case 1: no predictions
        precision = ones(2)

        if size(true_pos,2) == 0  # case 1a: no objects
            recall = ones(2)
        else                      # case 1b: there are objects
            recall = zeros(2)
            id1 = findall(x->x==1.0, true_pos[:,3])
            id2 = findall(x->x==-1.0, true_pos[:,3])

            if size(id1, 1) == 0 # case 1b1: no vortex objects
                recall[1] = 1.0
            end
            if size(id2, 1) == 0 # case 1b2: no antivortex objects
                recall[2] = 1.0
            end
        end

    else                         # case 2: there are predictions
        if size(true_pos,2) == 0 # case 2a: no objects
            id1 = findall(x->x==1.0, predicted_pos[:,3])
            id2 = findall(x->x==-1.0, predicted_pos[:,3])

            recall = ones(2)
            precision = zeros(2)
            if size(id1, 1) == 0 # case 2a1: no vortex predictions
                precision[1] = 1.0
            end
            if size(id2, 1) == 0 # case 2a2: no antivortex predictions
                precision[2] = 1.0
            end
        else                     # case 2b: there are objects
            precision = zeros(2)
            recall = zeros(2)

            idp1 = findall(x->x==1.0, predicted_pos[:,3])
            predicted_pos1 = predicted_pos[idp1,:]
            confidence1 = confidence[idp1]

            idt1 = findall(x->x==1.0, true_pos[:,3])
            true_pos1 = true_pos[idt1,:]

            idp2 = findall(x->x==-1.0, predicted_pos[:,3])
            predicted_pos2 = predicted_pos[idp2,:]
            confidence2 = confidence[idp2]

            idt2 = findall(x->x==-1.0, true_pos[:,3])
            true_pos2 = true_pos[idt2,:]

            predicted_poss = [predicted_pos1, predicted_pos2]
            confidences = [confidence1, confidence2]
            true_poss = [true_pos1, true_pos2]

            for (l, predicted_pos) in enumerate(predicted_poss)
                confidence = confidences[l]
                true_pos = true_poss[l]

                if size(true_pos, 1) == 0 # no object
                    recall[l] = 1.0
                    if size(predicted_pos, 1) == 0 # no prediction
                        precision[l] = 1.0
                    else                           # there is prediction prediction
                        precision[l] = 0.0
                    end
                elseif size(predicted_pos, 1) == 0 # there is object but no prediction
                    precision[l] = 1.0
                    recall[l] = 0.0
                else                               # there is object and prediction
                    already_detected = zeros(size(true_pos, 1))
                    id = sortperm(-confidence)
                    predicted_pos = predicted_pos[id,:]

                    true_positives = 0
                    false_positives = 0
                    for i = 1:size(predicted_pos, 1)
                        xB = predicted_pos[i,1]
                        yB = predicted_pos[i,2]
                        distance_list = zeros(size(true_pos, 1))

                        for j = 1:size(true_pos, 1)
                            xA = true_pos[j,1]
                            yA = true_pos[j,2]

                            distance_list[j] = distance(xA, xB, yA, yB)
                        end
                        jmin = argmin(distance_list)
                        dismin = distance_list[jmin]

                        if dismin <= distance_threshold
                            if already_detected[jmin] == 0
                                true_positives += 1
                                already_detected[jmin] = 1
                            else
                                false_positives += 1
                            end
                        else
                            false_positives += 1
                        end
                    end
                    total_positives = size(predicted_pos, 1) # same as: true_positives + false_positives
                    total_objects = size(true_pos, 1)

                    precision[l] = true_positives / total_positives
                    recall[l] = true_positives / total_objects
                end
            end
        end
    end
    return precision, recall
end;

function mean_precision_and_recall(model, x, y; confidence_threshold=0.5, distance_threshold=2.3, classes=false)
    """ Returns mean precision and recall over the whole data set """
    y_hat = nms(model(x), classes=classes)
    if classes
        precision, recall = unzip(precision_and_recall_classes.(eachslice(y_hat, dims=4), eachslice(y, dims=4); confidence_threshold=confidence_threshold, distance_threshold=distance_threshold))
    else
        precision, recall = unzip(precision_and_recall.(eachslice(y_hat, dims=4), eachslice(y, dims=4); confidence_threshold=confidence_threshold, distance_threshold=distance_threshold))
    end
    return mean(precision), mean(recall)
end;

function precision_recall_curve(model, test_images, test_labels; distance_threshold=2.3, classes=false)
    """ Returns precision and recall for 10 different confidence thresholds, the optimal confidence threshold, and the maximum F1 score """
    if classes
        return precision_recall_curve_classes(model, test_images, test_labels; distance_threshold=2.3)
    else
        conf_thres = range(0.05, stop=0.95, length=10)
        precisions, recalls = [], []
        for i in conf_thres
            prec, recall = mean_precision_and_recall(model, test_images, test_labels; confidence_threshold=i, distance_threshold=distance_threshold)
            append!(precisions, prec)
            append!(recalls, recall)
        end
        score = f1score.(precisions, recalls)
        max_id = argmax(score)
        return precisions, recalls, conf_thres[max_id], score[max_id]
    end
end;

function precision_recall_curve_classes(model, test_images, test_labels; distance_threshold=2.3)
    """ Returns precision and recall for 10 different confidence thresholds, the optimal confidence threshold,
        and the maximum F1 score separately for each class """
    conf_thres = range(0.05, stop=0.95, length=10)
    precs1, recalls1, precs2, recalls2 = [], [], [], []
    for i in conf_thres
        prec, recall = mean_precision_and_recall(model, test_images, test_labels; confidence_threshold=i, distance_threshold=distance_threshold, classes=true)
        append!(precs1, prec[1])
        append!(recalls1, recall[1])
        append!(precs2, prec[2])
        append!(recalls2, recall[2])
    end
    f11 = f1score.(precs1, recalls1)
    f12 = f1score.(precs2, recalls2)
    max_id1 = argmax(f11)
    max_id2 = argmax(f12)
    opt_conf_thres = [conf_thres[max_id1], conf_thres[max_id2]]
    f1_max = [maximum(f11), maximum(f12)]
    return [precs1, precs2], [recalls1, recalls2], opt_conf_thres, f1_max, [precs1[max_id1], precs2[max_id2]], [recalls1[max_id1], recalls2[max_id2]]
end;
