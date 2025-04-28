// translation_example.cpp
// ENG-FR translation example
// Download data from here: https://download.pytorch.org/tutorial/data.zip
// https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html

#include <tt/data/dataset.h>
#include <tt/device.h>
#include <tt/exception.h>
#include <tt/grad_mode.h>
#include <tt/index.h>
#include <tt/nn/dropout.h>
#include <tt/nn/embedding.h>
#include <tt/nn/gru.h>
#include <tt/nn/linear.h>
#include <tt/nn/loss.h>
#include <tt/nn/module.h>
#include <tt/optim/adam.h>
#include <tt/scalar.h>
#include <tt/tensor.h>

#include <algorithm>
#include <cctype>
#include <format>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <ranges>
#include <regex>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace tinytensor;

namespace {

constexpr int MAX_LEN = 10;
constexpr int SOS_token = 0;
constexpr int EOS_token = 1;
const std::unordered_set<std::string> eng_prefixes = {
    "i am ",
    "i m ",
    "he is",
    "he s ",
    "she is",
    "she s ",
    "you are",
    "you re ",
    "we are",
    "we re ",
    "they are",
    "they re "
};

auto str_split(const std::string &s, char delim) -> std::vector<std::string> {
    std::vector<std::string> strings;
    for (const auto word : std::views::split(s, delim)) {
        strings.emplace_back(std::string_view(word.begin(), word.end()));
    }
    return strings;
}

struct Lang {
    Lang(const std::string lang)
        : name(lang) {}
    void add_sentence(const std::string &sentence) {
        for (const auto &word : str_split(sentence, ' ')) {
            add_word(std::string(std::string_view(word.begin(), word.end())));
        }
    }

    void add_word(const std::string &word) {
        if (!word2index.contains(word)) {
            word2index[word] = n_words;
            word2count[word] = 1;
            index2word[n_words] = word;
            ++n_words;
        } else {
            ++word2count[word];
        }
    }

    std::string name;
    int n_words = 2;
    std::unordered_map<std::string, int> word2index;
    std::unordered_map<std::string, int> word2count;
    std::unordered_map<int, std::string> index2word = {{0, "SOS"}, {1, "EOS"}};
};

auto normalize_string(const std::string &str) -> std::string {
    std::string s = str;
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    std::regex r_trim("([.!?])");
    std::regex r_non_char("[^a-zA-Z!?]+");
    s = std::regex_replace(s, r_trim, " $1");
    s = std::regex_replace(s, r_non_char, " ");
    return s;
}

struct StrPair {
    std::string s1;
    std::string s2;
};

auto read_lang_file(
    const std::string &file_dir,
    const std::string &lang1,
    const std::string &lang2,
    bool reverse = false
) -> std::tuple<Lang, Lang, std::vector<StrPair>> {
    std::ifstream file(std::format("{:s}/{:s}-{:s}.txt", file_dir, lang1, lang2));
    std::string line;
    std::vector<StrPair> str_pairs;
    while (std::getline(file, line)) {
        std::vector<std::string> s_pair = str_split(line, '\t');
        if (s_pair.size() != 2) {
            TT_EXCEPTION("Expected line to be split by tab character");
        }
        str_pairs.emplace_back(normalize_string(s_pair[0]), normalize_string(s_pair[1]));
    }
    if (reverse) {
        for (auto &str_pair : str_pairs) {
            std::swap(str_pair.s1, str_pair.s2);
        }
        return {Lang(lang2), Lang(lang1), str_pairs};
    } else {
        return {Lang(lang1), Lang(lang2), str_pairs};
    }
}

auto filter_pairs(std::vector<StrPair> &&pairs) -> std::vector<StrPair> {
    auto starts_with = [&](const std::string &str) -> bool {
        for (const auto &prefix : eng_prefixes) {
            if (str.starts_with(prefix)) {
                return true;
            }
        }
        return false;
    };
    auto filter_pair = [&](const StrPair &pair) -> bool {
        return str_split(pair.s1, ' ').size() < MAX_LEN && str_split(pair.s2, ' ').size() < MAX_LEN
               && starts_with(pair.s2);
    };
    std::vector<StrPair> filtered_pairs;
    for (const auto &pair : pairs) {
        if (filter_pair(pair)) {
            filtered_pairs.push_back(std::move(pair));
        }
    }
    return filtered_pairs;
}

auto prepare_data(const std::string &file_dir, const std::string &lang1, const std::string &lang2, bool reverse = false)
    -> std::tuple<Lang, Lang, std::vector<StrPair>> {
    auto [input_lang, output_lang, pairs] = read_lang_file(file_dir, lang1, lang2, reverse);
    std::cout << std::format("Read {:d} sentence pairs", pairs.size()) << std::endl;
    auto filtered_pairs = filter_pairs(std::move(pairs));
    std::cout << std::format("Trimmed to {:d} sentence pairs", filtered_pairs.size()) << std::endl;
    std::cout << "Counting words..." << std::endl;
    for (const auto &pair : filtered_pairs) {
        input_lang.add_sentence(pair.s1);
        output_lang.add_sentence(pair.s2);
    }
    std::cout << "Counted words:" << std::endl;
    std::cout << std::format("{:s} - {:d}", input_lang.name, input_lang.n_words) << std::endl;
    std::cout << std::format("{:s} - {:d}", output_lang.name, output_lang.n_words) << std::endl;
    return {std::move(input_lang), std::move(output_lang), std::move(filtered_pairs)};
}

auto indexes_from_sentences(const Lang &lang, const std::string &sentence) -> std::vector<int> {
    std::vector<int> indexes;
    for (const auto &word : str_split(sentence, ' ')) {
        indexes.push_back(lang.word2index.at(word));
    }
    return indexes;
}

auto tensor_from_sentence(const Lang &lang, const std::string &sentence) {
    std::vector<int> indexes = indexes_from_sentences(lang, sentence);
    indexes.push_back(EOS_token);
    return Tensor(indexes, kCPU);
}

auto get_dataloader(
    const std::string &file_dir,
    const std::string &lang1,
    const std::string &lang2,
    bool reverse,
    int batch_size
) {
    auto [input_lang, output_lang, pairs] = prepare_data(file_dir, lang1, lang2, reverse);

    auto n = static_cast<int>(pairs.size());
    Tensor input_ids = zeros({n, MAX_LEN}, TensorOptions().dtype(kDefaultInt));
    Tensor target_ids = zeros({n, MAX_LEN}, TensorOptions().dtype(kDefaultInt));

    int idx = 0;
    for (const auto &[inp, tgt] : pairs) {
        std::vector<int> inp_ids = indexes_from_sentences(input_lang, inp);
        std::vector<int> tgt_ids = indexes_from_sentences(output_lang, tgt);
        inp_ids.push_back(EOS_token);
        tgt_ids.push_back(EOS_token);
        input_ids[{idx, indexing::Slice(0, inp_ids.size())}] = Tensor(inp_ids, kCPU);
        target_ids[{idx, indexing::Slice(0, tgt_ids.size())}] = Tensor(tgt_ids, kCPU);
        ++idx;
    }
    data::TensorDataset train_data(input_ids, target_ids);
    data::DatasetView train_data_view(std::move(train_data));
    data::DataLoader train_dataloader(train_data_view, batch_size);
    return std::make_tuple(input_lang, output_lang, pairs, train_dataloader);
}

class EncoderRNN : public nn::Module {
public:
    EncoderRNN(int input_size, int hidden_size, double p = 0.1)
        : embedding(input_size, hidden_size),
          gru(hidden_size, hidden_size, nn::GRUOptions{.batch_first = true}),
          dropout(p) {
        register_module(embedding);
        register_module(gru);
        register_module(dropout);
    }

    [[nodiscard]] auto name() const -> std::string override {
        return "EncoderRNN";
    }

    [[nodiscard]] auto forward(Tensor input) -> std::tuple<Tensor, Tensor> {
        Tensor embedded = dropout.forward(embedding.forward(input));
        auto [output, hidden] = gru.forward(embedded);
        return {output, hidden};
    }

private:
    nn::Embedding embedding;
    nn::GRU gru;
    nn::Dropout dropout;
};

class BahdanauAttention : public nn::Module {
public:
    BahdanauAttention(int hidden_size)
        : Wa(hidden_size, hidden_size), Ua(hidden_size, hidden_size), Va(hidden_size, 1) {
        register_module(Wa);
        register_module(Ua);
        register_module(Va);
    }

    [[nodiscard]] auto name() const -> std::string override {
        return "BahdanauAttention";
    }

    [[nodiscard]] auto forward(Tensor query, Tensor keys) -> std::tuple<Tensor, Tensor> {
        Tensor scores = Va.forward(tanh(Wa.forward(query).expand(keys.shape()) + Ua.forward(keys)));
        scores = scores.squeeze(2).unsqueeze(1);
        Tensor weights = softmax(scores, -1);
        Tensor context = matmul(weights, keys);
        return {context, weights};
    }

private:
    nn::Linear Wa;
    nn::Linear Ua;
    nn::Linear Va;
};

class AttnDecoderRNN : public nn::Module {
public:
    AttnDecoderRNN(int hidden_size, int output_size, double p = 0.1)
        : embedding(output_size, hidden_size),
          attention(hidden_size),
          gru(2 * hidden_size, hidden_size, nn::GRUOptions{.batch_first = true}),
          linear(hidden_size, output_size),
          dropout(p) {
        register_module(embedding);
        register_module(attention);
        register_module(gru);
        register_module(linear);
        register_module(dropout);
    }

    [[nodiscard]] auto name() const -> std::string override {
        return "AttnDecoderRNN";
    }

    [[nodiscard]] auto
        forward(Tensor encoder_outputs, Tensor encoder_hidden, std::optional<Tensor> target_tensor = std::nullopt)
            -> std::tuple<Tensor, Tensor, Tensor> {
        int batch_size = encoder_outputs.size(0);
        Tensor decoder_input =
            zeros({batch_size, 1}, TensorOptions().dtype(kDefaultInt).device(encoder_outputs.device()))
                .fill_(SOS_token);
        Tensor decoder_hidden = encoder_hidden;
        TensorList decoder_outputs;
        TensorList attentions;

        for (int i : std::views::iota(0, MAX_LEN)) {
            auto [decoder_output, _decoder_hidden, attn_weights] =
                forward_step(decoder_input, decoder_hidden, encoder_outputs);
            decoder_hidden = _decoder_hidden;
            decoder_outputs.push_back(decoder_output);
            attentions.push_back(attn_weights);

            const auto get_decoder_input = [&]() -> Tensor {
                if (target_tensor.has_value()) {
                    // Teacher forcing
                    return target_tensor.value()[{indexing::Slice(), i}].unsqueeze(1);
                } else {
                    // Without teacher forcing
                    return decoder_output.argmax(-1).detach();
                }
            };
            decoder_input = get_decoder_input();
        }

        return {log_softmax(cat(decoder_outputs, 1), -1), decoder_hidden, cat(attentions, 1)};
    }

    [[nodiscard]] auto forward_step(Tensor input, Tensor hidden, Tensor encoder_outputs)
        -> std::tuple<Tensor, Tensor, Tensor> {
        Tensor embedded = dropout.forward(embedding.forward(input));

        Tensor query = hidden.permute({1, 0, 2});
        auto [context, attn_weights] = attention.forward(query, encoder_outputs);
        Tensor input_gru = cat({embedded, context}, 2);

        auto [output, h] = gru.forward(input_gru, hidden);
        hidden = h;
        output = linear.forward(output);

        return {output, hidden, attn_weights};
    }

private:
    nn::Embedding embedding;
    BahdanauAttention attention;
    nn::GRU gru;
    nn::Linear linear;
    nn::Dropout dropout;
};

auto train_epoch(
    data::DataLoader<data::TensorDataset<Tensor, Tensor>> &dataloader,
    EncoderRNN &encoder,
    AttnDecoderRNN &decoder,
    optim::Adam &encoder_optimizer,
    optim::Adam &decoder_optimizer,
    Device device,
    const nn::NLLLoss &criterion
) -> double {
    double total_loss = 0;
    for (auto [input_tensor, target_tensor] : dataloader) {
        input_tensor = input_tensor.to(device);
        target_tensor = target_tensor.to(device);

        encoder_optimizer.zero_grad();
        decoder_optimizer.zero_grad();

        auto [encoder_outputs, encoder_hidden] = encoder.forward(input_tensor);
        auto [decoder_outputs, decoder_hidden, attentions] =
            decoder.forward(encoder_outputs, encoder_hidden, target_tensor);

        Tensor loss = criterion.forward(decoder_outputs.flatten(0, 1), target_tensor.flatten());
        total_loss += loss.item<double>();
        loss.backward();

        encoder_optimizer.step();
        decoder_optimizer.step();
    }
    return total_loss / dataloader.size();
}

void train(
    data::DataLoader<data::TensorDataset<Tensor, Tensor>> &dataloader,
    EncoderRNN &encoder,
    AttnDecoderRNN &decoder,
    Device device,
    int n_epochs,
    double lr = 3e-4
) {
    optim::Adam encoder_optimizer(encoder.parameters_for_optimizer(), lr);
    optim::Adam decoder_optimizer(decoder.parameters_for_optimizer(), lr);
    nn::NLLLoss criterion;

    for (int epoch : std::views::iota(0, n_epochs)) {
        double loss =
            train_epoch(dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, device, criterion);
        std::cout << std::format("Epoch: {:d}, loss: {:f}", epoch, loss) << std::endl;
    }
}

auto evaluate(
    EncoderRNN &encoder,
    AttnDecoderRNN &decoder,
    Device device,
    const std::string &sentence,
    const Lang &input_lang,
    const Lang &output_lang
) -> std::vector<std::string> {
    const autograd::NoGradGuard guard;
    Tensor input_tensor = tensor_from_sentence(input_lang, sentence).to(device).unsqueeze(0);

    auto [encoder_outputs, encoder_hidden] = encoder.forward(input_tensor);
    auto [decoder_outputs, decoder_hidden, attentions] = decoder.forward(encoder_outputs, encoder_hidden);

    Tensor decoder_ids = decoder_outputs.argmax(-1).flatten();
    std::vector<std::string> decoded_words;
    for (const auto &idx : decoder_ids) {
        auto idx_item = idx.item<int>();
        if (idx_item == EOS_token) {
            decoded_words.emplace_back("<EOS>");
            break;
        }
        decoded_words.push_back(output_lang.index2word.at(idx_item));
    }
    return decoded_words;
}

void evaluate_randomly(
    EncoderRNN &encoder,
    AttnDecoderRNN &decoder,
    Device device,
    const Lang &input_lang,
    const Lang &output_lang,
    std::vector<StrPair> &pairs,
    int n
) {
    auto rng = std::default_random_engine{};
    std::ranges::shuffle(pairs, rng);
    for (int i : std::views::iota(0, n)) {
        const auto &pair = pairs.at(static_cast<std::size_t>(i));
        std::cout << std::format("> {:s}", pair.s1) << std::endl;
        std::cout << std::format("= {:s}", pair.s2) << std::endl;
        std::vector<std::string> output_words = evaluate(encoder, decoder, device, pair.s1, input_lang, output_lang);
        std::cout << "< ";
        for (const auto &word : output_words) {
            std::cout << word << " ";
        }
        std::cout << std::endl;
    }
}

}    // namespace

constexpr int BATCH_SIZE = 64;
constexpr int HIDDEN_SIZE = 128;
constexpr int n_epochs = 20;

#ifdef TT_CUDA
constexpr Device device = kCUDA;
#else
constexpr Device device = kCPU;
#endif

// ./translation_example file_dir eng fra
int main(int argc, char *argv[]) {
    if (argc != 4) {
        TT_EXCEPTION("Usage: ./translation_example file_dir input_lang output_lang");
    }

    const std::string file_dir = argv[1];    // NOLINT(*-pointer-arithmetic)
    const std::string lang1 = argv[2];       // NOLINT(*-pointer-arithmetic)
    const std::string lang2 = argv[3];       // NOLINT(*-pointer-arithmetic)

    auto [input_lang, output_lang, pairs, dataloader] = get_dataloader(file_dir, lang1, lang2, true, BATCH_SIZE);
    EncoderRNN encoder(input_lang.n_words, HIDDEN_SIZE);
    AttnDecoderRNN decoder(HIDDEN_SIZE, output_lang.n_words);

    encoder.to(device);
    decoder.to(device);

    train(dataloader, encoder, decoder, device, n_epochs);
    evaluate_randomly(encoder, decoder, device, input_lang, output_lang, pairs, 10);
}
