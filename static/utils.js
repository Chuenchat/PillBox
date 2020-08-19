
// file access
var dataset_path = "static/data/train_real_04"
var images_path = dataset_path + "/images/"
var label_path = dataset_path + "/label.csv"
var library_path = "static/data/pill_library"
// var image_index = 11

// get file list to label
// var files = null;
// $.ajax(
//   {
//     url: images_path,
//     dataType: "json",
//     async: false,
//     success: function (data) {
//       files = data;
//     }
//   });

function decodeHtml(html) {
  var txt = document.createElement("textarea");
  txt.innerHTML = html;
  return txt.value;
}

// working modes
var mode = "m1"

// GUI
var curr_h = 100
var curr_w = 100
var cv_image = null
var blank = null
var cv_cross1 = null
var cv_cross2 = null
var cv_draw = null

// store
// {
//   image_id: {
//     file_name:
//     data: {
//       pill_id: {
//         x: 0;
//         y: 0;
//         w: 0;
//         h: 0;
//         shape: 0;
//         libreary_id: 0;
//       }
//     }
//   }
// }
// [sample_id, library_id, image_id, image_name, pill_id, pill_shape, bbox_X, bbox_Y, bbox_W, bbox_H]
// var label_data = new Array

// https://stackoverflow.com/questions/28782074/excel-to-json-javascript-code
selected_ids = [8, 11, 12, 14, 15, 17, 23, 24, 53]
shape_by_ids = []
lib_path = library_path + "/data.xlsx"
lib_data = null
count = 0
function load_xlsx(url) {
  // read xlsx
  /* set up XMLHttpRequest */
  var oReq = new XMLHttpRequest();
  oReq.open("GET", url, true);
  oReq.responseType = "arraybuffer";

  oReq.onload = function (e) {
    var arraybuffer = oReq.response;

    /* convert data to binary string */
    var data = new Uint8Array(arraybuffer);
    var arr = new Array();
    for (var i = 0; i != data.length; ++i) arr[i] = String.fromCharCode(data[i]);
      var bstr = arr.join("");

    /* Call XLSX */
    var workbook = XLSX.read(bstr, {
      type: "binary"
    });

    /* DO SOMETHING WITH workbook HERE */
    var first_sheet_name = workbook.SheetNames[0];
    /* Get worksheet */
    var worksheet = workbook.Sheets[first_sheet_name];
    // console.log(XLSX.utils.sheet_to_json(worksheet, { raw: true }));
    lib_data = XLSX.utils.sheet_to_json(worksheet, { raw: true })
    var lib_imgs = [];
    for (var i in lib_data) {
      if (selected_ids.includes(lib_data[i]['id'])) {

        shape_by_ids.push(lib_data[i]['shape'])

        img_id = "c_" + (count + 1)
        crop_size = 700
        show_size = 150

        // trust god, love god
        if (count == 0) {
          var c_1 = new Image();
          c_1.src = library_path + "/images/" + lib_data[i]['path'] + ".jpg"
          c_1.onload = () => {
            sx = (c_1.width - crop_size) / 2;
            sy = (c_1.height - crop_size) / 2;
            sw = crop_size;
            sh = crop_size;
            dx = 0;
            dy = 0;
            dw = show_size * 2
            dh = show_size
            document.getElementById("c_1").getContext("2d").drawImage(c_1, sx, sy, sw, sh, dx, dy, dw, dh)
          }
        }
        if (count == 1) {
          var c_2 = new Image();
          c_2.src = library_path + "/images/" + lib_data[i]['path'] + ".jpg"
          c_2.onload = () => {
            sx = (c_2.width - crop_size) / 2;
            sy = (c_2.height - crop_size) / 2;
            sw = crop_size;
            sh = crop_size;
            dx = 0;
            dy = 0;
            dw = show_size * 2
            dh = show_size
            document.getElementById("c_2").getContext("2d").drawImage(c_2, sx, sy, sw, sh, dx, dy, dw, dh)
          }
        }
        if (count == 2) {
          var c_3 = new Image();
          c_3.src = library_path + "/images/" + lib_data[i]['path'] + ".jpg"
          c_3.onload = () => {
            sx = (c_3.width - crop_size) / 2;
            sy = (c_3.height - crop_size) / 2;
            sw = crop_size;
            sh = crop_size;
            dx = 0;
            dy = 0;
            dw = show_size * 2
            dh = show_size
            document.getElementById("c_3").getContext("2d").drawImage(c_3, sx, sy, sw, sh, dx, dy, dw, dh)
          }
        }
        if (count == 3) {
          var c_4 = new Image();
          c_4.src = library_path + "/images/" + lib_data[i]['path'] + ".jpg"
          c_4.onload = () => {
            sx = (c_4.width - crop_size) / 2;
            sy = (c_4.height - crop_size) / 2;
            sw = crop_size;
            sh = crop_size;
            dx = 0;
            dy = 0;
            dw = show_size * 2
            dh = show_size
            document.getElementById("c_4").getContext("2d").drawImage(c_4, sx, sy, sw, sh, dx, dy, dw, dh)
          }
        }
        if (count == 4) {
          var c_5 = new Image();
          c_5.src = library_path + "/images/" + lib_data[i]['path'] + ".jpg"
          c_5.onload = () => {
            sx = (c_5.width - crop_size) / 2;
            sy = (c_5.height - crop_size) / 2;
            sw = crop_size;
            sh = crop_size;
            dx = 0;
            dy = 0;
            dw = show_size * 2
            dh = show_size
            document.getElementById("c_5").getContext("2d").drawImage(c_5, sx, sy, sw, sh, dx, dy, dw, dh)
          }
        }
        if (count == 5) {
          var c_6 = new Image();
          c_6.src = library_path + "/images/" + lib_data[i]['path'] + ".jpg"
          c_6.onload = () => {
            sx = (c_6.width - crop_size) / 2;
            sy = (c_6.height - crop_size) / 2;
            sw = crop_size;
            sh = crop_size;
            dx = 0;
            dy = 0;
            dw = show_size * 2
            dh = show_size
            document.getElementById("c_6").getContext("2d").drawImage(c_6, sx, sy, sw, sh, dx, dy, dw, dh)
          }
        }
        if (count == 6) {
          var c_7 = new Image();
          c_7.src = library_path + "/images/" + lib_data[i]['path'] + ".jpg"
          c_7.onload = () => {
            sx = (c_7.width - crop_size) / 2;
            sy = (c_7.height - crop_size) / 2;
            sw = crop_size;
            sh = crop_size;
            dx = 0;
            dy = 0;
            dw = show_size * 2
            dh = show_size
            document.getElementById("c_7").getContext("2d").drawImage(c_7, sx, sy, sw, sh, dx, dy, dw, dh)
          }
        }
        if (count == 7) {
          var c_8 = new Image();
          c_8.src = library_path + "/images/" + lib_data[i]['path'] + ".jpg"
          c_8.onload = () => {
            sx = (c_8.width - crop_size) / 2;
            sy = (c_8.height - crop_size) / 2;
            sw = crop_size;
            sh = crop_size;
            dx = 0;
            dy = 0;
            dw = show_size * 2
            dh = show_size
            document.getElementById("c_8").getContext("2d").drawImage(c_8, sx, sy, sw, sh, dx, dy, dw, dh)
          }
        }
        if (count == 8) {
          var c_9 = new Image();
          c_9.src = library_path + "/images/" + lib_data[i]['path'] + ".jpg"
          c_9.onload = () => {
            sx = (c_9.width - crop_size) / 2;
            sy = (c_9.height - crop_size) / 2;
            sw = crop_size;
            sh = crop_size;
            dx = 0;
            dy = 0;
            dw = show_size * 2
            dh = show_size
            document.getElementById("c_9").getContext("2d").drawImage(c_9, sx, sy, sw, sh, dx, dy, dw, dh)
          }
        }
        if (count == 9) {
          var c_10 = new Image();
          c_10.src = library_path + "/images/" + lib_data[i]['path'] + ".jpg"
          c_10.onload = () => {
            sx = (c_10.width - crop_size) / 2;
            sy = (c_10.height - crop_size) / 2;
            sw = crop_size;
            sh = crop_size;
            dx = 0;
            dy = 0;
            dw = show_size * 2
            dh = show_size
            document.getElementById("c_10").getContext("2d").drawImage(c_10, sx, sy, sw, sh, dx, dy, dw, dh)
          }
        }

        count += 1
      }
      if (count >= 10) {
        break
      }
    }
  }
  oReq.send();
}
load_xlsx(lib_path)

function shift(image_index) {
  var shift_index = image_index;
  if (shift_index < 0) { shift_index += files.length }
    if (shift_index > files.length - 1) { shift_index -= files.length }
      return shift_index
  }

  function update_image() {

  // update image slider
  document.getElementById("main_image").src = images_path + "/" + files[image_index]
  document.getElementById("s_pre3").src = images_path + "/" + files[shift(image_index - 3)]
  document.getElementById("s_pre2").src = images_path + "/" + files[shift(image_index - 2)]
  document.getElementById("s_pre1").src = images_path + "/" + files[shift(image_index - 1)]
  document.getElementById("s_now0").src = images_path + "/" + files[image_index]
  document.getElementById("s_nex1").src = images_path + "/" + files[shift(image_index + 1)]
  document.getElementById("s_nex2").src = images_path + "/" + files[shift(image_index + 2)]
  document.getElementById("s_nex3").src = images_path + "/" + files[shift(image_index + 3)]

  // update info block
  document.getElementById('name').innerHTML = files[image_index];
  document.getElementById('number').innerHTML = image_index + 1 + " of " + files.length;
  document.getElementById('path').innerHTML = images_path;

  // update data
  document.getElementById("main_image").onload = function () {
    cv_image = cv.imread('main_image');
    curr_h = cv_image.size()['height']
    curr_w = cv_image.size()['width']
    blank = cv_image.clone();
    blank.setTo(new cv.Scalar(255, 255, 255, 0));
    cv_cross1 = blank.clone();
    cv_cross2 = blank.clone();
    cv_draw = blank.clone();
  }
}

// move to next/previous image by keyboard left/right arrows
document.addEventListener('keydown', function (event) {

  if (event.key == "ArrowLeft") {
    (image_index == 0) ? image_index = files.length - 1 : image_index -= 1;
    update_image()
    update_draw()
  } else if (event.key == "ArrowRight") {
    (image_index == files.length - 1) ? image_index = 0 : image_index += 1;
    update_image()
    update_draw()
  } else if (event.key == "d") {
    detect()
  } else if (event.key == "s") {
    save()
  } else if (event.key == "c") {
    clear_label()
  }
});

selected_c = 1
var c_ids = [];
$('.b_class').each(function () {
  c_ids.push(this.id);
});

function pick_c(clicked_id) {
  for (i in c_ids) {
    if (c_ids[i] == clicked_id) {
      document.getElementById(c_ids[i]).style.border = "15px solid #00b900";
      selected_c = i;
    } else {
      document.getElementById(c_ids[i]).style.border = "";
    }
  }
}

// move to next/previous image by clicking sliders
function pick_s(clicked_id) {
  switch (clicked_id) {
    case "s_pre3":
    image_index -= 3;
    break;
    case "s_pre2":
    image_index -= 2;
    break;
    case "s_pre1":
    image_index -= 1;
    break;
    case "s_nex1":
    image_index += 1;
    break;
    case "s_nex2":
    image_index += 2;
    break;
    case "s_nex3":
    image_index += 3;
    break;
  }
  if (image_index < 0) { image_index += files.length }
    if (image_index > files.length - 1) { image_index -= files.length }
      update_image()
    update_draw()
  }

// // function draw_circle(e) {

// //   px = e.offsetX
// //   py = e.offsetY

// //   if (e.button === 2) {
// //     var [r, g, b] = [0, 0, 0]
// //   } else {
// //     var [r, g, b] = colors[selected_c]
// //   }
// //   color = new cv.Scalar(r, g, b, 255)

// //   img_cv = cv.imread("canvas")
// //   cv.circle(img_cv, new cv.Point(px, py), 5, color, -1)

// //   cv.imshow('canvas', img_cv);
// //   img_cv.delete

// // }

prev_x = 0;
prev_y = 0;
function handleMouseMove(e) {

  curr_x = e.offsetX
  curr_y = e.offsetY
  var [r, g, b] = colors[selected_c]
  color = new cv.Scalar(r, g, b, 255)
  invis = new cv.Scalar(0, 0, 0, 0)

  cv.line(cv_cross1, new cv.Point(prev_x, 0), new cv.Point(prev_x, curr_h), invis, 2)
  cv.line(cv_cross1, new cv.Point(0, prev_y), new cv.Point(curr_w, prev_y), invis, 2)
  cv.line(cv_cross1, new cv.Point(curr_x, 0), new cv.Point(curr_x, curr_h), color, 2)
  cv.line(cv_cross1, new cv.Point(0, curr_y), new cv.Point(curr_w, curr_y), color, 2)

  cv.imshow('overlay_cross1', cv_cross1);

  prev_x = curr_x;
  prev_y = curr_y;
}

box_state = 0;
box_x1 = 0;
box_y1 = 0;
function handleMouseUp(e) {

  curr_x = e.offsetX
  curr_y = e.offsetY
  invis = new cv.Scalar(0, 0, 0, 0)

  // left click - draw box
  // other clicks - cancel box
  if (e.button === 0) {

    var [r, g, b] = colors[selected_c]
    color = new cv.Scalar(r, g, b, 255)

    if (box_state == 0) {

      cv.line(cv_cross2, new cv.Point(curr_x, 0), new cv.Point(curr_x, curr_h), color, 2)
      cv.line(cv_cross2, new cv.Point(0, curr_y), new cv.Point(curr_w, curr_y), color, 2)
      cv.imshow('overlay_cross2', cv_cross2);
      box_x1 = curr_x
      box_y1 = curr_y
      box_state = 1

    } else if (box_state == 1) {

      cv.line(cv_cross2, new cv.Point(box_x1, 0), new cv.Point(box_x1, curr_h), invis, 2)
      cv.line(cv_cross2, new cv.Point(0, box_y1), new cv.Point(curr_w, box_y1), invis, 2)
      cv.imshow('overlay_cross2', cv_cross2);

      x1 = Math.min(box_x1, curr_x)
      y1 = Math.min(box_y1, curr_y)
      x2 = Math.max(box_x1, curr_x)
      y2 = Math.max(box_y1, curr_y)

      if (!(files[image_index] in label_data)) {
        label_data[files[image_index]] = new Array
      }
      label_data[files[image_index]].push([x1, y1, x2, y2, Number(selected_c)])
      update_draw()
      box_state = 0

    }

  } else {

    if (box_state == 0) {

      survived_data = new Array
      for (data_index in label_data[files[image_index]]) {
        [x1, y1, x2, y2, c] = label_data[files[image_index]][data_index]
        if (!((x1 < curr_x && curr_x < x2) && (y1 < curr_y && curr_y < y2))) {
          survived_data.push([x1, y1, x2, y2, c])
        }
      }
      label_data[files[image_index]] = survived_data
      update_draw()

    } else if (box_state == 1) {
      cv.line(cv_cross2, new cv.Point(box_x1, 0), new cv.Point(box_x1, curr_h), invis, 2)
      cv.line(cv_cross2, new cv.Point(0, box_y1), new cv.Point(curr_w, box_y1), invis, 2)
      cv.imshow('overlay_cross2', cv_cross2);
      box_state = 0
    }
  }
}

document.getElementById("overlay_draw").addEventListener('mousemove', handleMouseMove);
document.getElementById("overlay_draw").addEventListener('mouseup', handleMouseUp);
document.getElementById("overlay_draw").addEventListener('contextmenu', function (e) {
  e.preventDefault();
});

function update_draw() {

  cv_draw = blank.clone();
  for (data_index in label_data[files[image_index]]) {
    [x1, y1, x2, y2, c] = label_data[files[image_index]][data_index]
    var [r, g, b] = colors[c]
    color = new cv.Scalar(r, g, b, 255)
    cv.rectangle(cv_draw, new cv.Point(x1, y1), new cv.Point(x2, y2), color, 2)
  }
  cv.imshow('overlay_draw', cv_draw);
}

// c = null
// function detect() {
//   var text = files[image_index]
//   $.ajax({
//     type: "POST",
//     url: '{{detect/}}',
//     data: { csrfmiddlewaretoken: '{{ csrf_token }}', text:text },
//     success: function callback(response) {
//       console.log("here i am.")
//       console.log(response)
//       c = response
//     }
//   })
//   console.log("c : " + c)

//  //  $.ajax({
//  //    type: "POST",
//  //    url: '{{ url 'detect/'}}',   
//  //    data: {csrfmiddlewaretoken: '{{ csrf_token }}',
//  //          text: "this is my test view"},   /* Passing the text data */
//  //    success:  function(response){
//  //    console.log(response);
//  //   }
//  // });

// }


function load() {

  load_path = "C:/Users/chunc/Downloads/label.csv"

  // read csv
  $.ajax({
    url: load_path,
    async: false,
    success: function (csvd) {
      label_data = $.csv.toArrays(csvd).slice(1,);
    },
    dataType: "text",
    complete: function () {
      // call a function on complete 
    }
  });

  // fine
  console.log("load")
}

// https://stackoverflow.com/questions/14964035/how-to-export-javascript-array-info-to-csv-on-client-side
function save() {

  var rows = []
  rows.push([
    "sample_id", "library_id", "image_id", "image_name", "pill_id",
    "pill_shape", "bbox_X", "bbox_Y", "bbox_W", "bbox_H"
    ])

  count_image = 0
  count_sample = 0

  for (name in label_data) {
    count_pill = 0
    for (i in label_data[name]) {
      [x1, y1, x2, y2, c] = label_data[name][i]
      library_id = selected_ids[c]
      shape = shape_by_ids[c]
      rows.push([count_sample, library_id, count_image, name, count_pill, shape, x1, y1, x2-x1, y2-y1])
      count_pill += 1
      count_sample += 1
    }
    count_image += 1
  }

  let csvContent = "data:text/csv;charset=utf-8,";

  rows.forEach(function (rowArray) {
    let row = rowArray.join(",");
    csvContent += row + "\r\n";
  });

  var encodedUri = encodeURI(csvContent);
  var link = document.createElement("a");
  link.setAttribute("href", encodedUri);
  link.setAttribute("download", "label.csv");
  document.body.appendChild(link); // Required for FF

  link.click(); // This will download the data file named "my_data.csv".

  // fine
  console.log("save")
}

function clear_label() {

  // clear cross2 layer
  if (box_state == 1) {
    cv.line(cv_cross2, new cv.Point(box_x1, 0), new cv.Point(box_x1, curr_h), invis, 2)
    cv.line(cv_cross2, new cv.Point(0, box_y1), new cv.Point(curr_w, box_y1), invis, 2)
    cv.imshow('overlay_cross2', cv_cross2);
    box_state = 0
  }

  // clear draw layer
  label_data[files[image_index]] = new Array
  update_draw()

  // fine
  console.log("clear")
}

// hsv to rgb
function mix(a, b, v) {
  return (1 - v) * a + v * b;
}

function HSVtoRGB(H, S, V) {

  var V2 = V * (1 - S);
  var r = ((H >= 0 && H <= 60) || (H >= 300 && H <= 360)) ? V : ((H >= 120 && H <= 240) ? V2 : ((H >= 60 && H <= 120) ? mix(V, V2, (H - 60) / 60) : ((H >= 240 && H <= 300) ? mix(V2, V, (H - 240) / 60) : 0)));
  var g = (H >= 60 && H <= 180) ? V : ((H >= 240 && H <= 360) ? V2 : ((H >= 0 && H <= 60) ? mix(V2, V, H / 60) : ((H >= 180 && H <= 240) ? mix(V, V2, (H - 180) / 60) : 0)));
  var b = (H >= 0 && H <= 120) ? V2 : ((H >= 180 && H <= 300) ? V : ((H >= 120 && H <= 180) ? mix(V2, V, (H - 120) / 60) : ((H >= 300 && H <= 360) ? mix(V, V2, (H - 300) / 60) : 0)));

  r = Math.round(r * 255)
  g = Math.round(g * 255)
  b = Math.round(b * 255)

  return [r, g, b]
}
var colors = []
for (var i = 0; i < selected_ids.length; i++) {
  var h = 360 * (i / selected_ids.length);
  var s = 1;
  var v = 1;
  colors.push(HSVtoRGB(h, s, v))
}
// console.log(colors)

// start with update
window.onload = function () {
  update_image()
  update_image.onload = function () {
    load()
    if (!(files[image_index] in label_data)) {
      label_data[files[image_index]] = new Array
    }
    update_draw()
  }
}
console.log("this is fine.");

// to do
// - save 
// - clear
// - open
// - pill detection
// - watershed