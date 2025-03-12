$(document).ready(function () {
  $("#send-btn").click(function () {
    var userMessage = $("#user-input").val();
    if (userMessage.trim() !== "") {
      $("#chat-box").append('<div class="user">' + userMessage + "</div>");
      $("#user-input").val("");

      $.ajax({
        url: "/get_response",
        type: "POST",
        contentType: "application/json",
        data: JSON.stringify({ message: userMessage }),
        success: function (response) {
          $("#chat-box").append(
            '<div class="bot">' + response.response + "</div>"
          );
          $("#chat-box").scrollTop($("#chat-box")[0].scrollHeight);
        },
      });
    }
  });

  $("#user-input").keypress(function (e) {
    if (e.which === 13) {
      $("#send-btn").click();
    }
  });
});
