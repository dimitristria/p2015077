// main.js

function animateApply(element) {
    element.innerHTML = "<span class='spinner-border spinner-border-sm' role='status' aria-hidden='true'></span>  Applying...";
}

function path_leaf(path) {
    return path.split("\\").pop().split("/").pop();
}

// // init the state from the input
// $(".image-checkbox").each(function () {
//     if ($(this).find('input[type="checkbox"]').first().attr("checked")) {
//         $(this).addClass('image-checkbox-checked');
//     } else {
//         $(this).removeClass('image-checkbox-checked');
//     }
// });

// // sync the state to the input
// $(".image-checkbox").on("click", function (e) {
//     $(this).toggleClass('image-checkbox-checked');
//     var $checkbox = $(this).find('input[type="checkbox"]');
//     $checkbox.prop("checked", !$checkbox.prop("checked"))

//     e.preventDefault();
// });

// mousedown timer (delayed activation)
// $(document).ready(function (e) {
//     $(".img-check").mousedown(function (e) {
//         clearTimeout(this.downTimer);
//         this.downTimer = setTimeout(function () {
//             alert('mousedown > 1100 sec');
//             $(this).toggleClass("test");
//         }, 1100);
//     }).mouseup(function (e) {
//         clearTimeout(this.downTimer);
//     });
// });
