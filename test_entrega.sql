--armado de base para test

select a.*,b.edad,b.fecha_alta,b.genero,b.tipo, 
          c.localidad, cocina, precio, c.latitud, c.longitud, fotos, premios, "Ir en pareja", "Ir con amigos", "Comer con buenos tragos", "Llevar extranjeros", "Escuchar música", "Comer sin ser visto", "Comer al aire libre", "Comer solo", "Reunión de negocios", "Salida de amigas", "Comer bien gastando poco", "Ir con la familia", "Comer tarde", "Comer sano ", "Merendar", "Comer mucho", "Ir con chicos", "American Express", "Cabal", "Diners", "Electrón", "Maestro", "Mastercard", "Tarjeta Naranja", "Visa",
       telefono,comida_oleo,servicio_oleo,ambiente_oleo
into test_completo_v1
from
ratings_test a,usuarios b,rest_campos_v2 c
where a.id_usuario=b.id_usuario
and cast(a.id_restaurante as text)=c.id_restaurante

--para momento de entrega


select a.*,b.edad,b.fecha_alta,b.genero,b.tipo, c.localidad, cocina, precio, c.latitud, c.longitud, fotos, premios, "Ir en pareja", "Ir con amigos", "Comer con buenos tragos", "Llevar extranjeros", "Escuchar música", "Comer sin ser visto", "Comer al aire libre", "Comer solo", "Reunión de negocios", "Salida de amigas", "Comer bien gastando poco", "Ir con la familia", "Comer tarde", "Comer sano ", "Merendar", "Comer mucho", "Ir con chicos", "American Express", "Cabal", "Diners", "Electrón", "Maestro", "Mastercard", "Tarjeta Naranja", "Visa",telefono,comida_oleo,servicio_oleo,ambiente_oleo into test_completo_v1 from test_entrega a left join usuarios b on ( a.id_usuario=b.id_usuario) left join rest_campos_v2  on (cast(a.id_restaurante as text)=c.id_restaurante)


