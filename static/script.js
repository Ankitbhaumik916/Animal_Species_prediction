$(document).ready(function () {
    $("#predictBtn").click(function () {
        const data = {
            hair: $("#hair").val(),
            feathers: $("#feathers").val(),
            eggs: $("#eggs").val(),
            milk: $("#milk").val(),
            airborne: $("#airborne").val(),
            aquatic: $("#aquatic").val(),
            predator: $("#predator").val(),
            toothed: $("#toothed").val(),
            backbone: $("#backbone").val(),
            breathes: $("#breathes").val(),
            venomous: $("#venomous").val(),
            fins: $("#fins").val(),
            legs: $("#legs").val(),
            tail: $("#tail").val(),
            domestic: $("#domestic").val(),
            catsize: $("#catsize").val()
        };

        $.ajax({
            url: "/predict",
            type: "POST",
            contentType: "application/json",
            data: JSON.stringify(data),
            success: function (response) {
                $("#predictionResult").text("Predicted Animal Class: " + response.prediction);
            },
            error: function (xhr) {
                const errMsg = xhr.responseJSON && xhr.responseJSON.error
                    ? xhr.responseJSON.error
                    : "Something went wrong!";
                $("#predictionResult").text("Error: " + errMsg);
            }
        });
    });
});
