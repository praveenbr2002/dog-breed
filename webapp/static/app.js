Dropzone.autoDiscover = false;

function init() {
    let dz = new Dropzone("#dropzone", {
        url: "/",
        maxFiles: 1,
        addRemoveLinks: true,
        dictDefaultMessage: "Some Message",
        autoProcessQueue: false
    });
    
    dz.on("addedfile", function() {
        if (dz.files[1]!=null) {
            dz.removeFile(dz.files[0]);        
        }
    });

    dz.on("complete", function (file) {
        //var url = "http://127.0.0.1:8000/classify/";
        $.ajax({
            type:'POST',
            url:'/classify/',
            data:{
                image_data: file.dataURL
            },
            success: function(data){
                if (data['bool'] == 1){
                    console.log(data['class']);
                    $("#error").hide();
                    $("#resultHolder").show();
                    $("#divClassTable").show();
                    $("#result").html(data.class);
                    for(var i = 0; i < 3; i++){
                        $("#score_"+String(i)).html(data.class_probability[i]);
                        $("#label_"+String(i)).html(data.class_names[i]);
                    }

                }
            }
          });
        
    });

    $("#submitBtn").on('click', function (e) {
        dz.processQueue();		
    });
}

$(document).ready(function() {
    console.log( "ready!" );   
    $("#error").hide();
    $("#resultHolder").hide();
    $("#divClassTable").hide();
    init();
});

