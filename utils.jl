# custom functions used in corresponding jupyter notebook

unzip(a) = map(x->getfield.(a, x), [1, 2]);

function convert_positions_to_labels(positions; grid_size=64, image_size=256)
    """ Returns 64x64x3xn_images label tensors computed from list of vortex positions """
    n_images = size(positions, 3)
    labels = zeros(grid_size, grid_size, 3, n_images)
    grid_cell = image_size ÷ grid_size
    max_objects = size(positions, 1)
    for n = 1:n_images
        for i = 1:max_objects
            if positions[i,1,n] > 0 # check first if object exists
                x = Int((positions[i,1,n] - 1) ÷ grid_cell + 1)
                y = Int((positions[i,2,n] - 1) ÷ grid_cell + 1)

                x_offset = (positions[i,1,n] - 1) / grid_cell + 1 - x
                y_offset = (positions[i,2,n] - 1) / grid_cell + 1 - y

                labels[x, y, 1, n] = 1.0
                labels[x, y, 2, n] = x_offset
                labels[x, y, 3, n] = y_offset
            end
        end
    end
    return labels
end;

function convert_label_to_position(label; threshold=0.5, grid_cell=4)
    """ Returns list of vortex positions computed from single 64x64x3 label tensor """
    positions = []
    grid_idx = findall(x->x>=threshold, label[:,:,1])
    confidence = zeros(length(grid_idx))
    for (i, id) in enumerate(grid_idx)
        confidence[i] = label[id[1],id[2],1]

        grid_offset_x = grid_cell * (id[1] - 1)
        grid_offset_y = grid_cell * (id[2] - 1)

        x = grid_offset_x + grid_cell * label[id[1],id[2],2] + 1
        y = grid_offset_y + grid_cell * label[id[1],id[2],3] + 1

        append!(positions, [[round(x), round(y)]])
    end
    return round.(Int, hcat(positions...)'), confidence
end;

function convert_labels_to_positions(labels; threshold=0.5, grid_cell=4)
    """ Returns list of vortex positions computed from a batch of 64x64x3xn_images label tensors """
    return unzip(convert_label_to_position.(eachslice(labels, dims=4), threshold=threshold, grid_cell=grid_cell))
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

function nms(probability_map; kernel=3)
    """ Non-max suppression to eliminate multiple detections of the same vortex """
    pad = (kernel - 1) ÷ 2
    pmax = NNlib.maxpool(probability_map[:,:,1:1,:], (kernel, kernel), pad=pad, stride=1)|>gpu
    keep = float.((pmax .== probability_map[:,:,1:1,:]))
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

""" Returns F1 score: harmonic mean between precsion and recall """
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

function mean_precision_and_recall(model, x, y; confidence_threshold=0.5, distance_threshold=2.3)
    """ Returns mean precision and recall over the whole data set """
    y_hat = nms(model(x))
    precision, recall = unzip(precision_and_recall.(eachslice(y_hat, dims=4), eachslice(y, dims=4); confidence_threshold=confidence_threshold, distance_threshold=distance_threshold))
    return mean(precision), mean(recall)
end;

function precision_recall_curve(model, test_images, test_labels; distance_threshold=2.3)
    """ Returns precision and recall for 10 different confidence thresholds, the optimal confidence threshold, and the maximum F1 score """
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
end;
