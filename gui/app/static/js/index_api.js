// Example starter JavaScript for disabling form submissions if there are invalid fields
(function () {
  'use strict'

  const urlBart = '/api/processing/bart'
  const urlBartAndBert = '/api/processing/all'
  const urlBert = '/api/processing/bert'

  function setRequiredField(inputId) {
    const inputField = document.getElementById(inputId)
    const liField = document.getElementById(inputId + '__li')
    if (inputField) {
      // inputField.required = true
      inputField.disabled = false
    }
    if (liField) {
      liField.classList.remove('disabled')
    }
  }

  function setDisabledField(inputId) {
    const inputField = document.getElementById(inputId)
    const liField = document.getElementById(inputId + '__li')
    if (inputField) {
      inputField.required = false
      inputField.disabled = true
    }
    if (liField) {
      liField.classList.add('disabled')
    }
  }

  function setDisplayNone(fieldId, isNone, text) {
    const field = document.getElementById(fieldId)
    if (field) {
      if (isNone === true) {
        field.classList.add('d-none')
      } else {
        field.classList.remove('d-none')
      }
      if (text) {
        field.innerHTML = text
      }
    }
  }

  function filledResultField(text, isHide) {
    const resultFieldDiv = document.getElementById('result_field__div');
    const resultFieldText = document.getElementById('result_field__value');
    if (resultFieldDiv) {
      if (isHide) {
        resultFieldDiv.classList.add('d-none')
      } else {
        resultFieldDiv.classList.remove('d-none')
      }
    }
    if (resultFieldText) {
      resultFieldText.innerHTML = text
    }
  }

  function getAbstractiveSummarizationParameters() {
      let input_text = document.getElementById('input_text')
      let max_length = document.getElementById('max_length')
      let no_repeat_ngram_size = document.getElementById('no_repeat_ngram_size')
      let num_beams = document.getElementById('num_beams')
      let repetition_penalty = document.getElementById('repetition_penalty')
      let temperature = document.getElementById('temperature')
      let top_k = document.getElementById('top_k')
      let top_p = document.getElementById('top_p')
      return {
          input_text: input_text.value || input_text.placeholder,
          max_length: max_length.value || max_length.placeholder,
          no_repeat_ngram_size: no_repeat_ngram_size.value || no_repeat_ngram_size.placeholder,
          num_beams: num_beams.value || num_beams.placeholder,
          repetition_penalty: repetition_penalty.value || repetition_penalty.placeholder,
          temperature: temperature.value || temperature.placeholder,
          top_k: top_k.value || top_k.placeholder,
          top_p: top_p.value || top_p.placeholder,
      }
  }

  const changeTypeFieldByMethod = () => {
    const selectMethod = document.getElementById('choose_method')
    if (selectMethod) {
      const options = selectMethod.options
      switch (options[selectMethod.selectedIndex].id) {
        case 'bart':
        case 'bart_and_bert': {
          setRequiredField('max_length')
          setRequiredField('no_repeat_ngram_size')
          setRequiredField('num_beams')
          setRequiredField('repetition_penalty')
          setRequiredField('temperature')
          setRequiredField('top_k')
          setRequiredField('top_p')
          setDisplayNone('text_muted_bart', false)
          setDisplayNone('result_error__text', true, '')
          setDisplayNone('button_summarize_loading', true)
          setDisplayNone('button_summarize', false)
          filledResultField('', true)
          break
        }
        case 'none_method':
        case 'bert': {
          setDisabledField('max_length')
          setDisabledField('no_repeat_ngram_size')
          setDisabledField('num_beams')
          setDisabledField('repetition_penalty')
          setDisabledField('temperature')
          setDisabledField('top_k')
          setDisabledField('top_p')
          setDisplayNone('text_muted_bart', true)
          setDisplayNone('result_error__text', true, '')
          filledResultField('', true)
          setDisplayNone('button_summarize_loading', true)
          setDisplayNone('button_summarize', false)
          break
        }
      }
    }
  }

  function validationSummarizeForm() {
    const form = document.getElementById('form_summarize');
    let check = true
    if (!form.checkValidity()) {
      check = false
    }

    form.classList.add('was-validated')
    return check
  }

  function errorResponse(text) {
    setDisplayNone('button_summarize', false)
    setDisplayNone('button_summarize_loading', true)
    setDisplayNone('result_error__text', false, text)
    return false
  }

  const sendSummarize = () => {
    setDisplayNone('button_summarize', true)
    setDisplayNone('button_summarize_loading', false)
    setDisplayNone('result_error__text', true, '')
    const checkValidation = validationSummarizeForm()
    if (checkValidation) {
      let data = [];
      let url = null
      const selectMethod = document.getElementById('choose_method')
      if (selectMethod) {
        const options = selectMethod.options
        switch (options[selectMethod.selectedIndex].id) {
          case 'bart': {
            data = getAbstractiveSummarizationParameters()
            url = urlBart
            break
          }
          case 'bart_and_bert': {
            data = getAbstractiveSummarizationParameters()
            url = urlBartAndBert
            break
          }
          case 'bert': {
            data = {
              input_text: document.getElementById('input_text').value
            }
            url = urlBert
            break
          }
          default:
            return errorResponse('Not choice method!')
        }
      } else {
        return errorResponse('Not choice method!')
      }

      axios.post(url, data)
          .then((response) => {
            setDisplayNone('button_summarize', false)
            setDisplayNone('button_summarize_loading', true)
            filledResultField(response.data.result, false)
            return response.data
          })
          .catch((error) => {
            return errorResponse(error.message)
          });
    } else {
      setDisplayNone('button_summarize', false)
      setDisplayNone('button_summarize_loading', true)
    }
  }

  changeTypeFieldByMethod()
  document.getElementById('choose_method').addEventListener('change', changeTypeFieldByMethod)
  document.getElementById('button_summarize').addEventListener('click', sendSummarize)
  filledResultField('', true)

  $('#count_text').html('0')
  document.getElementById('input_text').addEventListener('keyup', function() {
    let text_length = $(this).val().length
    $("#count_text").html(text_length)
  });
})()
