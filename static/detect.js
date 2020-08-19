function go_python() {

  // $.ajax({
  //   url: "detection.py",
  //   context: document.body
  // }).done(function () {
  //   alert('finished python script');;
  // })

  $.ajax({
    method: "GET",
    url: "type your python script path here",
    data: { "place": value },
    dataType: "text",
    success: function (result) {
      var data = JSON.parse(result);
      console.log(result);
    }
  });


  
}