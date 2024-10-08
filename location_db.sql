PGDMP  ;    6                |           location_details    17.0    17.0                0    0    ENCODING    ENCODING        SET client_encoding = 'UTF8';
                           false                       0    0 
   STDSTRINGS 
   STDSTRINGS     (   SET standard_conforming_strings = 'on';
                           false                       0    0 
   SEARCHPATH 
   SEARCHPATH     8   SELECT pg_catalog.set_config('search_path', '', false);
                           false                       1262    16389    location_details    DATABASE     �   CREATE DATABASE location_details WITH TEMPLATE = template0 ENCODING = 'UTF8' LOCALE_PROVIDER = libc LOCALE = 'Polish_Poland.1252';
     DROP DATABASE location_details;
                     postgres    false            �            1259    16391    cities    TABLE     �   CREATE TABLE public.cities (
    id integer NOT NULL,
    name character varying(255) NOT NULL,
    encoded_city double precision,
    city_avg_price double precision
);
    DROP TABLE public.cities;
       public         heap r       postgres    false            �            1259    16390    cities_id_seq    SEQUENCE     �   CREATE SEQUENCE public.cities_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 $   DROP SEQUENCE public.cities_id_seq;
       public               postgres    false    218                       0    0    cities_id_seq    SEQUENCE OWNED BY     ?   ALTER SEQUENCE public.cities_id_seq OWNED BY public.cities.id;
          public               postgres    false    217            �            1259    16400 	   districts    TABLE     �   CREATE TABLE public.districts (
    id integer NOT NULL,
    name character varying(255) NOT NULL,
    city_id integer NOT NULL,
    encoded_district double precision,
    district_avg_price double precision
);
    DROP TABLE public.districts;
       public         heap r       postgres    false            �            1259    16399    districts_id_seq    SEQUENCE     �   CREATE SEQUENCE public.districts_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 '   DROP SEQUENCE public.districts_id_seq;
       public               postgres    false    220                       0    0    districts_id_seq    SEQUENCE OWNED BY     E   ALTER SEQUENCE public.districts_id_seq OWNED BY public.districts.id;
          public               postgres    false    219            �            1259    16414    neighborhoods    TABLE     �   CREATE TABLE public.neighborhoods (
    id integer NOT NULL,
    name character varying(255) NOT NULL,
    district_id integer NOT NULL,
    encoded_neighborhood double precision,
    neighborhood_avg_price double precision
);
 !   DROP TABLE public.neighborhoods;
       public         heap r       postgres    false            �            1259    16413    neighborhoods_id_seq    SEQUENCE     �   CREATE SEQUENCE public.neighborhoods_id_seq
    AS integer
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;
 +   DROP SEQUENCE public.neighborhoods_id_seq;
       public               postgres    false    222                       0    0    neighborhoods_id_seq    SEQUENCE OWNED BY     M   ALTER SEQUENCE public.neighborhoods_id_seq OWNED BY public.neighborhoods.id;
          public               postgres    false    221            a           2604    16394 	   cities id    DEFAULT     f   ALTER TABLE ONLY public.cities ALTER COLUMN id SET DEFAULT nextval('public.cities_id_seq'::regclass);
 8   ALTER TABLE public.cities ALTER COLUMN id DROP DEFAULT;
       public               postgres    false    218    217    218            b           2604    16403    districts id    DEFAULT     l   ALTER TABLE ONLY public.districts ALTER COLUMN id SET DEFAULT nextval('public.districts_id_seq'::regclass);
 ;   ALTER TABLE public.districts ALTER COLUMN id DROP DEFAULT;
       public               postgres    false    219    220    220            c           2604    16417    neighborhoods id    DEFAULT     t   ALTER TABLE ONLY public.neighborhoods ALTER COLUMN id SET DEFAULT nextval('public.neighborhoods_id_seq'::regclass);
 ?   ALTER TABLE public.neighborhoods ALTER COLUMN id DROP DEFAULT;
       public               postgres    false    221    222    222                      0    16391    cities 
   TABLE DATA           H   COPY public.cities (id, name, encoded_city, city_avg_price) FROM stdin;
    public               postgres    false    218   }"                 0    16400 	   districts 
   TABLE DATA           \   COPY public.districts (id, name, city_id, encoded_district, district_avg_price) FROM stdin;
    public               postgres    false    220   M#                 0    16414    neighborhoods 
   TABLE DATA           l   COPY public.neighborhoods (id, name, district_id, encoded_neighborhood, neighborhood_avg_price) FROM stdin;
    public               postgres    false    222   s-                  0    0    cities_id_seq    SEQUENCE SET     ;   SELECT pg_catalog.setval('public.cities_id_seq', 6, true);
          public               postgres    false    217                       0    0    districts_id_seq    SEQUENCE SET     ?   SELECT pg_catalog.setval('public.districts_id_seq', 88, true);
          public               postgres    false    219                       0    0    neighborhoods_id_seq    SEQUENCE SET     D   SELECT pg_catalog.setval('public.neighborhoods_id_seq', 456, true);
          public               postgres    false    221            i           2606    16407    districts _district_city_uc 
   CONSTRAINT     _   ALTER TABLE ONLY public.districts
    ADD CONSTRAINT _district_city_uc UNIQUE (name, city_id);
 E   ALTER TABLE ONLY public.districts DROP CONSTRAINT _district_city_uc;
       public                 postgres    false    220    220            m           2606    16421 '   neighborhoods _neighborhood_district_uc 
   CONSTRAINT     o   ALTER TABLE ONLY public.neighborhoods
    ADD CONSTRAINT _neighborhood_district_uc UNIQUE (name, district_id);
 Q   ALTER TABLE ONLY public.neighborhoods DROP CONSTRAINT _neighborhood_district_uc;
       public                 postgres    false    222    222            e           2606    16398    cities cities_name_key 
   CONSTRAINT     Q   ALTER TABLE ONLY public.cities
    ADD CONSTRAINT cities_name_key UNIQUE (name);
 @   ALTER TABLE ONLY public.cities DROP CONSTRAINT cities_name_key;
       public                 postgres    false    218            g           2606    16396    cities cities_pkey 
   CONSTRAINT     P   ALTER TABLE ONLY public.cities
    ADD CONSTRAINT cities_pkey PRIMARY KEY (id);
 <   ALTER TABLE ONLY public.cities DROP CONSTRAINT cities_pkey;
       public                 postgres    false    218            k           2606    16405    districts districts_pkey 
   CONSTRAINT     V   ALTER TABLE ONLY public.districts
    ADD CONSTRAINT districts_pkey PRIMARY KEY (id);
 B   ALTER TABLE ONLY public.districts DROP CONSTRAINT districts_pkey;
       public                 postgres    false    220            o           2606    16419     neighborhoods neighborhoods_pkey 
   CONSTRAINT     ^   ALTER TABLE ONLY public.neighborhoods
    ADD CONSTRAINT neighborhoods_pkey PRIMARY KEY (id);
 J   ALTER TABLE ONLY public.neighborhoods DROP CONSTRAINT neighborhoods_pkey;
       public                 postgres    false    222            p           2606    16408     districts districts_city_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.districts
    ADD CONSTRAINT districts_city_id_fkey FOREIGN KEY (city_id) REFERENCES public.cities(id);
 J   ALTER TABLE ONLY public.districts DROP CONSTRAINT districts_city_id_fkey;
       public               postgres    false    4711    218    220            q           2606    16422 ,   neighborhoods neighborhoods_district_id_fkey    FK CONSTRAINT     �   ALTER TABLE ONLY public.neighborhoods
    ADD CONSTRAINT neighborhoods_district_id_fkey FOREIGN KEY (district_id) REFERENCES public.districts(id);
 V   ALTER TABLE ONLY public.neighborhoods DROP CONSTRAINT neighborhoods_district_id_fkey;
       public               postgres    false    4715    222    220               �   x��)nAEq�aZ�/'0H@�IH+f�f�1�4�O����ྎ��>jFYX����P�ڴb)�*_�zF#Sb1d���gK5��<����(�+[
b�Cuǔ
��v������lX**�*)@�V��{�G�0���mmMK��^@��3CJ�ș�������J��kF������NYU:~��K;�         
  x�m��n�H����B/`���/'	&;�d$3k �Z&lF�8����W�l���aK,vթS�N{�i���W�����8oZWJ��.W\4�4�b�|k�6�/����8�s�w���b�U���iTl����,Rc�6�����?-�𲒜o�U_���j�Z�N$�Z��\�i���G'�I��lZ(�{_C�O�_��d��0m���84�HBJb�K�v���IO�$+��8l��%��gQ*_�\����b+!ƒ�-��i;l���3�8�I���|�bJ>J��m5���%��KW�D�s���t9�V��.������ay?`
�2j� i+D�Q�zg�GJ���"�K�T(:����/䐈��~7��"��KN^xB��C��\Ճ�V��:mǛi^Nɹx�6$L�}���������ֵ�J5Qi���y�����b��N��a??�=|h5TP$@m1wPN8Nm�����8,���gM���1�.�K�Z2YZO����qZ��5Oέ�̫�uZ/"A *�c���K+Bv<	�t9¿Z]#�5���|���G҃">%-)����5��X�̻=y��Ͻ	�[u�S�笤�����]ʥ��47fu�R.��0h�q
�V+0N]��6Bco�7�����_?�� TR�	�2�@?r��V�5Fb�~?�>���a�� �ɲorE�g>����4��A�a�����_o���-�PЂ��pw�wD�]�4��z������(��ؓ+H2C�Zv!fϬِ�[N�#7��ZJ(>�Z���K�9ˮ�B��<��<���eMN�>����N ������jQ����M�q}�<�4�x0G�Q�ŝD)�Pn�6�y�ߍÑ3ǫW�~ج~����t.�-\A�U�M(�h$X�dC3o�_R�'ͨ@�LF$;H���{�~tO�X��'�j�dj�s�p�|K�6z��|�4�&����r$+����Gӽ' ������
UL:�s�-�P�$�@�,4��ƨ1gխ�O{T{=~�����d��$H�"�6�Ij���r��g�FH=����eA� N�	l���~�=���(��DӪ"�����"�ls}V���;�;l{����t'�
� 5s����˫E#�N�]o"ϻ�[d{�LoMmP��˦�+��� ��B��6x�*��<����2nڋ<!l��:N���X�əW����i�J 4�бb��  k��l���p�$E�.��=mo0OY��Z������3Iy%����^��չɵ�Q�Ȥh~�o��������a&r".u���C�I/�Iä�XX�����W������ϩj��x�S�P>���)@r����<V�Ɍ��+�i�����ooMV��za;Q��Y2ɉ��0� �)M��9��34J�'xAE���@�p4�"Ψ��_��mU�R�m�U���xžf�1|@���j [ "1n�f����_�'�V���d&9�xʫ�
+�p�s^�W:'#�z�h;ʭG}���y�c�9R�ajD����4�P���sXP����9*���io����DQ�˥0<z]uX�*���
�������2�O��I����f]���y�]c7�P�>��E�E�)�%3ʀ��)�Ϟ��-�� p�` 6`��L �1�\����O�y:�o�'	D���j*���0���26$	?��v{�$8e��`'�N��%^[*:�9�!육Jê��� �I������t7.�
űN�uکbH��=
/�b%�w��A׳�"J~�������V�����1�f��0ީs�ebq�Z'kl�������V\�d>2|c��ح��otv}�~��g� ]�rF]=;�@tI#|������%תD b�8x�D�le�>��*���̙���j����6��[��g6>E�)��'�����S4�P1[��a�?v���+��5jp1�lt{zV��x��-�QFQ�.��yu�0655Kє��~��<OY�>QDqa��{*�t�HL���q~�>=R�{o��\�3Qw+60?H��~x؍�� ��'��|�#�GH��`TpxYl	��v��I���xb8��������Z�b��՗�fI=��J��1"�&G9� �̎�������y�!0�YU��[�$�cp�6��p��]�����v$79�)f���=˅-��oǇ�#�9��q%�sIУ�6�.��� h{�6��kυ��F�
ڏPq�P�gv0W���S�O
����������N�!�P��iK��p�����O8!&�� l��t�ƻ[
u3��8� �U����I�Вm�x�bYi����判��d(H}��qv���c3Y�&����O�X܌�B�5�^�p�ˤ\���Skq�*��\�]Ǖ��,1c�&���0�k����T���E5�s�q�^t�Vև^ov�E4Y����gyb�#kiv���l�Y�"-z�
 ��>������I�,��l�Y��1IA�	��2n��ԐC='�K���c��8�`+j���;��{�sk��������&�            x�m��rǑ��{�7������CK^�E�!���w2f�!�� p���d� �-Y֮E�*+��7��o>lo��ov߶M���4Mu꺮�:�)7���Ԗ�O��i��Mj�y|�]��oo�Mj����q���}�J75)ujk����r���ͻ���~�|���<ϥ桖�o������8M���\�&7��w������OO����&��L���ԍ�=6}?���e��M�uS�w��7�M�8������aΩ֦���~����8oj����Ύ7Ǧ�bƱG��4|�Զ�Ŀ�7�0��������xw|��뻙k�<6�ﻮ-��.����|:^=�o������0���a���/�e���sߏi37�>�>�.�7�K5��4��������s*]��)m���p��=�w��������~Ί��Min�a*�ܥ�/Ӧ�����~C�K4�qR�Ɣ��?��M��O�O����a�kf�=����/��ۙQ�v�=�@(ɦ����������h|p���2�w�e觼�p��P���>�|ǫ���1�X����S�B�&~����̹J����awu��:��=���EjQ�:v�8�����G���������4��yp�!�L��͘���CB�9�����?�_�7������s��TY��.��%cC�7��^=^�~<�?O�LRlv`wC�{�4zo��ȭ���.�~n�]��#�k�JI��;�ڤ���%�s,I#&�5o������vQE��ğՌ�Už��ScP�a����m�-�(!:���(y�sWѦ)��I�y��]�O�7�y5^dľ
��@1s'ܤ������w�`�Y�w+�S��}ɥIC����g$�Iy������:tU�T�)OH�^R͕Mm��z��J�I����8V-���`h��j��v��;�6��Q������ZD?�%���8)����xyخa��������N�5O]�??5����' i��̽`y V�jl�s�`����&���3\Tp�����HJ9�(,�$���f !�����6y��i����c��L 2j=`\�ӸA/>\O��'tӯ����4�%(�C��f�~֤�?NO�9���L���3�"�i�b���7ր��߶��tl�pq@�.��t���G8	��9�xw���->�W�:�}X����0s��Ji~�}H~��xw ��q������*��s-��NT��������c�a��6�	�4�A�,����i3����v����+�����3I|�����|:�b�~�B��Ș6�����NE)6�<����p�8��&�"��elX�4p2�k��m�E�X��x��ְ1c�  �	�H�/pW�a��W7{���I��o�;��8�?&��e��P���7�OۻŅ᎒��[$N|��d�a����������r报i�@^z(F���V@�M߰�x�~^of�е���s3 ��=]�i�K�`�o�_����N  ѕ�=`���J˾�N 1J�k�������.Ն$�~�H�\.!/�a{���ۮ�~L J��g̨D���s�P/ij�o�8��oΌ����l�pJfn~�=폧�����+�:Ķ���t8�M�V�wvt��`L���Yz]9�£�]ޔ�����a{�-�;ĺJ�r�-|	��i�h�8m�x�cD��=�Z�n�� ���*� ��Z7����p\� 5-�dF�76S�`(kW��~�l�{��d��*Ԏ��F(��9�me�Ma�O��M��ǟ��PE�:���#pUe�P����pz\�Fl
0�"{��@![|aE#@Bp��g.��O�����ׅ��H�[1!�%#��2/�%���ԫ��QH�+@�3����&���W�~8�9��,n�I�a���e�6�k~�^��U,��\�)������mS��-��vs�Q�(�\|�1A&���&�B�[qS��a�p��:^Y.H��<�Ns�0 ��x=7��><m�A���{�%P<s��mh����3+r��'ne�X5������9c��+x�ۀN�$�БE�E�9ά��{�3ԌS�aXuC����t�. R�@��-	�E���=s6x�7������x�8T�.��0`�&�u,�PN�w��l���7k�0�P�SWE�3Г����s����9�&�w���zI��bf08@��nu���?���2)�b���Im�Bm8C�!�K�����q�3���:|@�¸\�fL�T_�7w?�v���>
+�|�B�<%X ��Am�����1����U+.����C��V����`��ã��@��!�2p�Im%u;5(� X�l���S����^�E\V�0��ڢ��"X4qΑe�����M0�r�.5�Pv4*���qu�v@tC�����D�t�F3U�ARi[y��5g���b|�&���M>�7c|����-
�E�M��o��s�|I�(�T1G ���f1�P�H<�v��.�n|,�@.&\)$w:+��<�7����+Z��xy�p��8:5k�x��} Mw�0���SU�4K����ĕ�4���m����d�XH���g���_��U5J�L@|��u��`2BU�L%N�7���5a(� ���E����)��&�� Vt8h	��4L�������	�Ǩ���A��F ��_�=̉K͆���wpB�����r���a��m�Lz��`W ��X����p��Y/�p��pÕ���ld>! gi��t�+߮�w2xǕiHI׈��鄞�2��ܭ��J�*�C����.�dŃc��x�<���9���o�g�(vԹ�2��"s7i5[T����q!+l�Dt5����O�hA�p����<�d�t����&Q�!��qUf�2g�d��_����ѣ �Nh3�P���6�g�~:ɇ.><q��.׻�f��(Z?�1��d����SU��q��\���VW��"
�CxX�^܋d�F����gS��x{<�)។[f;���`dKx�?a�f���0��]�'�+a	|Ҡ`�c�~޾<�	�a�//�q}DLx�=������'����9���1�^V=�mv�+B�Dl�|�����5�.˺1ܤ����e� 	��"�X_��C�	}E?�o�x���TK#���_�OW�>û�N�N1�g��kJm*�A4��#�D�?��*�z��z�U�5�6���4 ��O�F �<�ޠ�|�S��7�Ըm���/7�W9�F��a�<^�+�]�rH���a�����*��+B�b�`�F7YBϱ=������f��&�c��3C�Z{�6/��rM�馊�B�$�a_1�����Hz<�n�k��@!]�Ͱ$��ž%�k�~��H%s�`���b���p���ށ@qB�}ZĚD:N���+�]9K`Q��B����nN���PU8+O'+P+�Q��C�<s�����tw���
s=������X��%2��)@�~�}�>/2'�<�4ʣ���hC�pbEQ���߭�֔�o�DJ�-�m��A.��hZ�v	�-2q�!-��*�!��6��G�g,ys��<�v��$vkt4���ú&��0��m����v�`T�t��+0gj���́q�L��hw���+3/�]>����.������v�T�w���E�Y�0DS����~*C��!PL�ƈS�, ��63�X�!����I�s��t�L�(m�~P������^ީY @bNw��W��m�:�^��|���\2���B��nvcU^�_dȕ�@�q���-���=ho_���p^��O7���{Sa@�,�@����8s�N��j�E�E��۳��t�̌�����L��ҟ���_9���U(��h�[�m2LG;Q�z蚏�[ �9�^Uz� f^8��B+��ת�*В��D�j�HT ChL�p:����p����u���
j� �HC�o`{��翼o����ý�8�ň;����JҦ�y	8��\�U�\3�!Cӈ�    A�X�ByWv���,+R���ZW�}6�^á�B�h�l�R�(�ge���wQ���#>��N���TZiR�AU��ecCdu�����>r�����b�Z��@��~���N�OKJ/�+�zq��9�NNQJ��Xu�l�E31�����&-1����/��/��U�6�J�"���_�����S����� �M�U�`�A�Е{N�1?=�ٞ&�Qx�Z=q6�fT��I�u\^��=��V�6JS�D�5�ZD�,K��#v��lO
��병S+[���Ͳ8�Io����m�@#콚�����Q4 ˛t������7�Ҹ=�H8p��$��FB�ȬG%S���/]���BO�{��oja�	�Cg�h�&����勉ύ�T#O.����o�Y6��g�>��;m"�Q���v1&yh�%�%~vn����TYZM$�㆓�'�S�-S���H��S_���\�;�P0]f>D~���`��ʹ��r�в�ܤF4�(���nt�'��[��o��O�/�������q�r���%Cw�z���YI�iPn��,W~��,}���f�Gú���.�h�vz��̋K�!��aYJ%�\ֺ���Z���5��cMm~�}n��2f�]����Q�[B	s����\�!-����Ŵ�4�C2J@.���~ws���=)�|�/
��&譀�x Rހ�N�
�����]�������G-�a�Z�Ҁb��v�_��O/jk�	������ż#$�|C�U�{���#lVO�,O����}�>
1Hk,�Hkj�
�Ƒ{ˍ(87��	m �U2���n,������b,�62l� ��s�������b `:^�׿6ð��4m�-!��37�E^DnBb�ܞ��W�Q����%�b�m&�\9!�Xk������ MPDLċHiI[X��u@LAYX^�_�ZN�� P9d6��#VĪ�j��k��ޝ��LS9�`})�U:B+)/:���)��`H:γ)E�J�Y1i��2k��H{��$�p��D^�7?HM�L4�z�\��gq>��Z;�MT�LFU#�Gs�P ���UVVv���А���RV v���O��K�?p���'w�`��#�*
gg*<�8rt���������&$��l7�h��}I J�ƕn�a�l#vh���i�|m����
��ʺ򶋏��W�����laJ�Gm�k�u ���cl~��z�ߜ�>[��b��P�Hd1G�C��qz�z�9���>�b��Þ��dQ6e���^��*M0���1f8��\�"�E$}���=,ng:�0�\c�H��:�F�5���ǧ`�Yv�lٰ�d����
��:4�K�<f��i���ۻ� %���	�.�:�i�y�+��Zvo�OWK� �`���L:�S�U�td�����G���d��\Kna�X���v���`O���u�J���퀏6��l\*���H�>�q{�}�.�E����85cf������p���oO��'����`�L�AP3M���&�����j�.���q4� zt_�q@�7����O�`��3"����+c�:ï^�0����'K�gg<�O����Q;�_&�h$� �Z����z~�ao��lO@�[7�:�������*�UF��?D�2���$k����,�R!"S�"�8�g}�{�a)��oI��
��@�'�z0"+��e���*q�ǧ/Y�3�88o81��)3�t�����d���{4��,����GL�
4����Frw���ʬG�$�(� GT i�55��h��w����ѻ��LKW��`���LV�`�)��N����`"(��=���.��\z���M2;�f�����Q���'� <��`I�moB��ȽYK��� �ɶ�� `4��d���/6<���w�W�d�R	� �5������Y���nI%�) �E@.�q��]���~�:��Yɀ"U۱l�j�VB4�_�v���&_p��aI!�W��p��BTXU�_�?�ڡ��q�nE�2���b�(��.jY��S1/�K�Dk"�>g�.MD	_L ��c1o��>�/oEl>d��g�������
Ul��Os����f:����)�.�r-�KlOeN��֎Q�bj���=�p{�y�-[�rh�H�*�8ZTˢ�gwXi~UB������Qه�{����(�9����]Y�%p?@�F�Ȭ(�_�;uTkvi�M�跁5b��
��ӑ`�E�[7e�a@�d����ʸ��Y����t(�=�٩ha�F��d�"L��׶�ٞ�7.1s�h�h��`�֕�+�a�fIO�O��u/:�j��R��k-��� �Y��[�����m��V�1j���CBMJ]a3҅�=¯L홁��k�5R��sc���{�g���
��f�G ��$8��ܟ���W����Ǚ��k+D׼�ϧݧ�A߱��٬y�u�g4b��������ۯ��~�JH��f��#�J^x�]U�.�:j�:yS!�L� 	�kc�`k�$�=�,�K��ͯ�4�@
D a�st �֟M�'�,�M$�oﶗ_�����`�ȏ��T�8EY� �mN�.�PgC�d ̈XK~��� ������t\�A��}
r�W���lG�l��+��9�nl������Ldݡ<(���韓Kf��EL��g#>--�E�c[����測Hއ��d���b�\Мjo�-Z���A)rg�"$�,w�s��y�+<0K��]˽����&�#��q؞XHK�������&�����C�c����9��Af1��u�������C��2�'��s�J����FT�����_�����S���T���̾�뵓qto8����2|�V���7���Ċ- ɠ'�12�<�.���L��R�":5#���ǡ�F�/wQ9�5�uo>�X�8ά/&e�����l.�5������Yh+���R��,1r]j���a�jgI��W�RmZ=�M[�,�6g��?D8���y�+@�gM��n��w�ge5N��,��.�y������i�5���O���Y���vqzEf�'�V4r�z�E?�q����E�g�A��S�!b���ɵ�_������`�~��FY"���]@�YMkxN*�fm��x��P T�D�(�f��۷m·�E�u*��y��o?�-b���}tg��T��C�8НI���"��aʄ��$�(��+>sח������wc}��,C�&������닽�J��Jd��OP����N�[Ν<�W�"�pJ�����3�^ľ���]nټG�m���gYQ�A�Q�^��l�%�)�{8B���J�Ib�E�n�qB=Lg��u\���]0�=[P:�o�s�H�dL:�:A�J?]4��1��`lhء��4��{ԏ�p�a�~QΜ?��a٦V�T߿I��Few�O�GE���)�h4�:�*�������{a�o�!R`�+z����[�O簿C�x�@�0_9����M�#�����%�85
��N��y0��*��	x|�k�*CU�g���M僝� �ĭ��]�ƊS���򶧅�-�i�d���^�{"g�{@�
�?c��!���ۃ=:ϖ�[Ƶ(��}�	����w�x������a��a y(�̻��	{�&޸�b�R�����iu���R�>@�L\�.�:Ir�}x�;H���*;�Y6Q@!7ɶЯ��r�UH��*�d`P�hq �I�sޔ)1�0C3K��4��L���司0��x�w�7�
�F �����܋����)#���G��2���F[��e���:%�|=�An����3���������-/Y�6T;�.�억N�h T�6kq�94�a��'\��K� /���k�.�hɎ�L�~okF�x�6���� +����˴��)�0�q`!{0D����+�`�7�md����;?�"�wܲ�6�G�/bM]4*�mo.���\�49r~�1��L��b.M�2�hG���w	����%�y�CLih�fa�t�k    ��΋ߑ���k0v� !�	ͦ�
3h�N"�Z���w��m��ŢX�o��9F
��������~��qY $ف:���tB���� �n���瘵8���mɣ�&Ǭ�%H[K��!�'od�<-B�h�Q�la�\T��O�F��e)|��Kc��Шqs���7;�k�U�fn����E~�p��#���]��D�\g�[������utHEu���Zi#$Z�
U@!l�e�%�9~�~Y�Ś�oc�h���!Q����q-�}����eC�!��h�Xp�0�Ks߿��w���A��ۜj�N�[��an�I$NE����d{�"G�B֨�7�,���w3��;"�9G{�~ѤH�:�F<Ų�!��:�fPo���\k�k�,��l�w���G�[�m�s�\F����9==�X!?�g��A����R��N9��������-����A��iFՌkZ1��f���U�ˡ��˱�w�c`�
אq̣�K��8�v�x����϶�ʹD�v0����������c�}1)bx[{����:��G���o���j�=]Ӕ#J
�@WLd�m�eU_��\�jy'���}ޓ���:�n��C?Ok9�XFb���V��2m��p*,�x�,���e��B,%�ԥ�ǱkV��F������㲹j��`�/t�a���Š�5�y�?���� ��\���2(�c�G���dl�}3'^P�Tb��9Nb��hң7�Ɗɦ���� k�րJ��`�t��5ى�T�̪�:�B�b�F�ܮci$�������׃+��V+��:�1ES9�5�����aU����E ��^�`9|�D��5h��|	�R���W_����pi���t�=\=�T0ō.�G�I[Ǭ�9�������ݺ&��
����<SL:��F��,)K��ٽ&˯^�	9�[��Zt��M)A���m�m>�J��T�������o���Oh:�Ç�7��g�1����zL�u�L�����no*�a{�ǕY�V�ݎ)��!W?nOw�9�E�tf[��b��dE��}�-u/�����X��=zbV?���ˮ��%V��7p.����P��e�2*���*�88oX�{��~|�8m�Xz P�daET�h���ӂbksF��K��$l�8�����H����Đ�����U���_s����5�r>��y�4	u�I�!���L��<��ܯ#�ǥ������ɘ�w{H��R��l��n82�r���E��lZ��-�:6Sq6�1�~�E���:����).X�V$Qt>�������<̤ �h�Q���l!�|���-���l"�χ�������� ���n���|�Gv>)a��X�l��W0��L��M�v�/���k�E)9��u���uc���a�9:�\�r-�T���t<�s1cv,;94��Y��w�~�<��e�پ��Y4O��r2,�<�o��/~�mV�J�)�1/df(1�d���Z�� Ɉs[�ʳ�/��Ӹ�w55��NO��9��~��(1j�4{�w;eb����Eǿ�u6�u�!��*���VЗ���v�d���46�1��|&���|0�~z|���!�bA?9�3:7g֛5�����ì��͜s�֘o��x��j�%6�Me��g2�ї�^<�؀G��ߐ�BL��{z���ʗ��*9�k����	Z��t���5�9޶��k.������[��v٫U_S�s��z�Y*\}@ė%!�_�O��2��8&��	�hT+)F`P�ᛍo�|<�1N'��뫫xi�e�7�4�t��2��o��l���ڮh2!�]:���m�s,�&�L��NV/�d���t���6ל��r,�;�p��_8mv�L�tCj��v���Y�� �tu�5C'|�`�d��a��BR\�)����Υ��-Gf~Iıjh�vZ�����q�0k7�lY)곱3��i�u��y��8���c2Fdȅ�S���1�Y�ȺtQ����y��ܶ�F�{dIm~��m��KL���]�{F�lri�Y�9,&��7$�=%��	��M�D��9똞Vq�H.9���\�26?�!�8��c��+�Qo��ٟ�fvh�k�������٤��dj��5��T��ln�`�ʢ�� ��B�����v���m��Ά �;�g����nS�ћ.A��j?�S��X}��>�����Y��@����*��ɩ�R��ok&#l�Y�I�)͑�#�i��/�2���x�ʖl���+�5�;T�����b����
¦ER��P=�>�2�hŵ���Ox�Ú3:ש}��4�au;������?����|R���"ٟ�����4V�HƷ�r�N?���~�N]������gTq@�%f����P��y�,k(���_��-���*�	�3O�;1=��dG���-.�ҟ��8��b��<Ũ;(��a�{��	��f<�����̦�G``2R��3���5���>�߸���4uN�֘���R#���9|��������^�F\�T��+%��U��7B�ƨ��y7���
�c��vNR�I��� ���ެ�Gc�Y��j�j⬔�N�؛�W.�u��ޢ���e㊪9>#b_1�Y>l����z؞�����v��馞�%㽭n6Wkc������S��ќ���F��B�X� �bzS߼=�����-P�X�L�C�E��Zq$��q�q�����E%�{}��D����BPL����h�}٘�Ofk��r�G�k\��~��#ű*�6���>ZEch�wd:?Q@_��d���8"}4+G|`[l��TzJ�SC��J?G��u0Ob��$X M��5��4~~�Pd�9E<0�����0�}�,��-����Y�0�����3�
�Β[�ck��}]R�����J�i5q�1��k�_3�;���/7wǛ���h:�O�[;�lj/V����`�Ӫ���߾�}�8 ���D����7�vF���byn�Y(0��V�=b1�h�? ;j�C�\�2L{�h1Yb2��=ߺ����co|�W;�/����Co1%�m������{�H�L���8��Ų�zd�g=4��o�>8���	�f:K��h�Ƈ�o7��k����������f@���^�>����xX�&�������4��j�vƍ(�EC��T��y�>%`WD�H���5���e�!���$K��9	/I�1یBٺhm���zy�����FrH���h�4���4Qu�ߑ���E��yv0E��K���+s�{���o|���/�2k��u�(����"=�t\�j�o9"��'�u_X��� qs�eՕWA@P�Wʂ�����e�H�,�v���1�&��m�zѰ�8�� 9mJ�B��G��kF��ᕲ|��_�ϾQڻGu��;�Pͻw�Y�p����+|
0w������;��>�fL����po"x7��^:�"��	� d��x��r.�5��Y�B�A�sT2���%
���x����"V��~�6�-f�|�����nχ�5+K���PHr"g��߇Cl���|.�A��~\F��+8)d�<����|v��c��^| 4X�W���mX�N׀{�� �h�6rw��G����c�>|��������k�vF%���xδ��''��1�sD/VM�� �C�x7�X5ǋ�ˌ��&D�AT��(�:5eW�}��G�^k�Sj*{������ ��I�S��� �%��]�O�ٹ�i`o�$����懮����cJͿ�X[�����сIN������Jp^����*���+�Ǖ���{6%���Gͦ�a7Xm�G_U�X����W��G'K�\�Q��1�������Mf�ܴ��[�J�ۚ���}50���u^7���9q�������%�]k�%[�!��~K�>�V�-᫬}?�*�ѭ0�U	s}��S��#��e�4-/�� q��	pj�t\<�*��]�o)����5L�B,Oj��^�ַ�|�֖JDL]��W���	��u�>j�3��@�GrW��#���v�Y��K��m�ēX*mF)�s/xM�����=^�C���/���?���� �  Tj芏�ٷ6�&�w���NU:o�0�#&�f]r�=��s��_bQ1ұO���;��w��?\�|%[;'A�x�L�] �-Gr�92e��KLA}�U%^^�['��r'ܤ!��rgȝ���{]����6ކ�8ؤFo�����S���w ���1N�O~煩K�?Kȋcվ38�̚����z<�:N�(��F�&3��KxoJ<�B���Ldy��1�m����lGDqw�quU����aϡ�e2��#�O}ɇ��x�7�{����#�s��F#�xy*��|���3ǔ�	Y��	!�>X����٥�E��lnǾ�>�5;����4Ǌ�л��j�(����C�H���g���1bQ�Y���Aϯ��KZ��.����#f�Td�c���M�~��G���k|��-Q�$�2zvJ�G������Ŝ���\u���bN�Z��j�Z���(v���Ow0�%�3C���UfKIgI;�M���)8ʧ�2���,7c������Yⴗ��z�'�"E�̶Y�u��q���M霩t����xڇs��Z�z�Q�e�}�cwbc��D����Q&�����樆�Ȁ����v������<���Ģ{5��YD���lҎQ�v�����B�:_J�̝��D�>w�3�����7�u��z���ǜ��jm�Y�^�2��# _n����(v�6�D,��}�o���X�����v\��A܁��V�����k7����Z�     