int main(int argc, char* argv[]) {
    {% for snippet in snippets %}
    {{snippet.render()}}
    {% endfor %}
    return 0;
}